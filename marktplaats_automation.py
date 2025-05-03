#!/usr/bin/env python3
import asyncio
import traceback
import os
import json
import glob
import hashlib
import logging
from typing import List, Optional
from playwright.async_api import async_playwright, Page
import time
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("marktplaats.log")],
)
logger = logging.getLogger(__name__)

HEADLESS = True
FAKE_POSTING = True
DELAY = 4000


class MarktplaatsAutomation:
    def __init__(self, headless: bool = HEADLESS, slow_mo: int = 50):
        self.headless = headless
        self.slow_mo = slow_mo

        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Set paths relative to the script directory
        self.cookies_path = os.path.join(script_dir, "marktplaats_cookies.json")
        self.images_folder = os.path.join(script_dir, "images")

        # Initialize browser variables
        self.browser = None
        self.context = None
        self.page = None

    async def __aenter__(self):
        """Context manager entry point."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=self.headless, slow_mo=self.slow_mo
        )

        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720}
        )

        # Load cookies if they exist
        if os.path.exists(self.cookies_path):
            with open(self.cookies_path, "r") as f:
                cookies = json.load(f)
                await self.context.add_cookies(cookies)

        self.page = await self.context.new_page()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def _handle_cookie_dialog(self):
        """Handle cookie accept dialog if it appears."""
        cookie_button = self.page.locator('button[title="Accepteren"]')
        if await cookie_button.count() > 0:
            await cookie_button.click()
            await self.page.wait_for_timeout(
                DELAY
            )  # Wait for cookie banner to disappear
            logger.info("Cookie accept clicked; waiting done")

    async def _handle_modal_dialog(self):
        """Handle modal dialog with 'Bedankt, ik snap het!' button if it appears."""
        modal_button = self.page.locator(
            'button.hz-Button.hz-Button--primary:has-text("Bedankt, ik snap het!")'
        )
        if await modal_button.count() > 0:
            logger.info("Modal found, clicking 'Bedankt, ik snap het!' button")
            await modal_button.click()
            await self.page.wait_for_timeout(DELAY)
        modal_button = self.page.locator(
            'button.hz-Button.hz-Button--primary:has-text("Verder")'
        )
        if await modal_button.count() > 0:
            logger.info("Modal found, clicking 'Verder' button")
            await modal_button.click()
            await self.page.wait_for_timeout(DELAY)

    async def _extract_messages_from_group(self, message_group):
        """Extract messages from a message group element."""
        messages = []

        # Get the date heading for this group
        date_heading = message_group.locator(".DateHeading-module-root")
        date = "Unknown date"
        if await date_heading.count() > 0:
            date = await date_heading.inner_text()

        # Get all message items in this group
        messages_list = message_group.locator("ol.Messages-module-listRoot")
        if await messages_list.count() > 0:
            message_items = messages_list.locator("li.Messages-module-listItem")
            messages_count = await message_items.count()

            if messages_count > 0:
                # Process each message in the group
                for j in range(messages_count):
                    message_item = message_items.nth(j)

                    # Extract message text
                    message_text_element = message_item.locator(
                        ".MessageElement-module-body"
                    )
                    if await message_text_element.count() > 0:
                        message_text = await message_text_element.inner_text()

                        # Extract timestamp
                        timestamp = await self._extract_timestamp(message_item)

                        # Determine sender
                        side = await self._determine_message_sender(message_item)

                        # Add message to list
                        message_info = {
                            "side": side,
                            "text": message_text,
                            "_timestamp": timestamp,
                            "_date": date,
                        }
                        messages.append(message_info)

        return messages, date

    async def _extract_timestamp(self, message_item):
        """Extract timestamp from a message item element."""
        timestamp = "Unknown time"

        try:
            # Try to find a more specific timestamp element first
            timestamp_element = message_item.locator(
                ".MessageElement-module-meta > .hz-Text.hz-Text--bodySmall.u-colorTextSecondary"
            ).first

            if await timestamp_element.count() > 0:
                timestamp = await timestamp_element.inner_text()
        except Exception as e:
            logger.info(f"Error getting timestamp: {e}")

            # Alternative approach - get all elements and use the first one
            try:
                all_timestamp_elements = await message_item.locator(
                    ".hz-Text.hz-Text--bodySmall.u-colorTextSecondary"
                ).all()

                if all_timestamp_elements and len(all_timestamp_elements) > 0:
                    timestamp = await all_timestamp_elements[0].inner_text()
            except Exception as e2:
                logger.info(f"Alternative timestamp extraction also failed: {e2}")

        return timestamp

    async def _determine_message_sender(self, message_item):
        """Determine if a message is from the user or the other participant."""
        message_class = await message_item.get_attribute("class")

        if message_class:
            if "Messages-module-listItem_from_otherparticipant" in message_class:
                return "other"
            elif "Messages-module-listItem_from_me" in message_class:
                return "me"

        return "unknown"

    def _generate_chat_id(self, title, messages):
        """Generate a hash ID for a chat based on its content."""
        # Clean messages to just side and text for consistent hashing
        clean_messages = [
            {"side": msg["side"], "text": msg["text"]} for msg in messages
        ]

        # Generate content string for hashing
        chat_content = title + "".join(
            [f"{msg['side']}:{msg['text']}" for msg in clean_messages]
        )

        # Generate hash and take first 8 characters
        chat_hash = hashlib.md5(chat_content.encode("utf-8")).hexdigest()
        return chat_hash[:8]

    async def save_cookies(self):
        """Save browser cookies to a file for future sessions."""
        try:
            cookies = await self.context.cookies()
            with open(self.cookies_path, "w") as f:
                json.dump(cookies, f)
            logger.info(f"Cookies saved to {self.cookies_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save cookies: {str(e)}")
            return False

    async def login(self, username: str, password: str) -> bool:
        try:
            # Navigate to main page first
            await self.page.goto("https://www.marktplaats.nl/", wait_until="commit")
            await self.page.wait_for_timeout(DELAY)

            # Accept cookies if the dialog appears
            await self._handle_cookie_dialog()
            logger.info("Cookie button not found; all good")

            # Check if we're already logged in by looking for the login button
            login_button = self.page.locator('a[data-role="login"]')

            if await login_button.count() == 0:
                logger.info("Already logged in!")
                return True

            # Otherwise, click on the login button to go to the login page
            await login_button.click()

            # Fill in login form
            await self.page.fill("#email", username)
            await self.page.fill("#password", password)
            logger.info("Login form filled")

            # Click login button and wait for navigation
            await self.page.click(
                'button.hz-Button.hz-Button--primary:has-text("Inloggen met je e-mailadres")'
            )
            logger.info("Login button clicked")

            # Wait for user to complete phone verification (when URL becomes marktplaats.nl)
            logger.info("Waiting for phone verification to complete...")
            logger.info(
                "Please complete the verification in the browser window (if it is there)."
            )

            # Give user time to complete verification (max 5 minutes)
            max_wait = 300  # 5 minutes
            start_time = time.time()
            while time.time() - start_time < max_wait:
                if "https://www.marktplaats.nl/" == self.page.url:
                    logger.info("Verification successful!")
                    await self.save_cookies()
                    return True
                else:
                    logger.info(f"Current URL: {self.page.url}")
                await asyncio.sleep(1)

            logger.error("Verification timeout reached. Please try again.")
            return False

        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            return False

    async def _get_personal_messages(self) -> bool:
        """
        Navigate to the personal messages page by clicking on the 'Berichten' link.
        If messages don't appear within 5 seconds, reload the page to fix the bug.

        Returns:
            bool: True if successfully navigated to messages page, False otherwise
        """
        try:
            # Make sure we're on the main page
            await self.page.goto("https://www.marktplaats.nl/", wait_until="commit")
            await self.page.wait_for_timeout(DELAY)
            logger.info("Navigated to Marktplaats homepage")

            # Accept cookies if the dialog appears
            await self._handle_cookie_dialog()

            # Find and click the "Berichten" (Messages) link
            messages_link = self.page.locator(
                'a.hz-Link[data-role="messaging"][title="Berichten"]'
            )
            logger.info("Searching for messages link... done")

            if await messages_link.count() == 0:
                logger.warning(
                    "Could not find the 'Berichten' link. Are you logged in?"
                )
                return False

            # Click on the link to go to the messages page
            await messages_link.click()
            logger.info("Clicked on messages link")

            # Verify we're on the messages page
            if "messages" in self.page.url:
                logger.info("Successfully navigated to messages page")

                # Wait for messages to appear (maximum 5 seconds)
                max_wait_time = 5

                # Look for conversation list items or other message UI elements
                message_selectors = [
                    # Conversation list item
                    ".ConversationsList-module-listItem",
                    "li.ConversationsList-module-listItem",
                    ".ConversationItem-module-root",
                    # Message content
                    ".ConversationItem-module-body",
                    ".ConversationItem-module-latestMessageWrapper",
                ]

                selector = ", ".join(message_selectors)
                messages_container = self.page.locator(selector).first

                logger.info(
                    f"Waiting up to {max_wait_time} seconds for messages to appear..."
                )

                success = False
                for i in range(10):
                    # Wait for any of the message-related elements to be visible
                    try:
                        await messages_container.wait_for(
                            state="visible", timeout=max_wait_time * 1000
                        )
                        logger.debug(
                            "Messages or message UI elements loaded successfully"
                        )

                        # Count the conversation items found
                        conversation_items = self.page.locator(
                            ".ConversationItem-module-root"
                        )
                        count = await conversation_items.count()
                        if count > 0:
                            logger.debug(f"Found {count} conversation items")
                            success = True
                        else:
                            logger.info("No conversation items found, reloading page")
                            await self.page.reload()
                    except Exception as e:
                        logger.info(
                            f"Messages not appearing, reloading page... {str(e)}"
                        )
                        await self.page.reload()
                        await self.page.wait_for_timeout(DELAY)

                if not success:
                    logger.warning("Messages are not appearing after multiple attempts")
                    return False
                return True

            else:
                logger.warning(f"Navigation failed. Current URL: {self.page.url}")
                return False

        except Exception as e:
            logger.error(f"Error accessing messages: {str(e)}")
            return False

    async def read_messages(self, *, send_id=None, send_msg=None) -> dict:
        """
        Read all messages from conversations in the messages page.
        For each conversation:
        1. Click on it
        2. Handle any modal by clicking "Bedankt, ik snap het!" button if it appears
        3. Parse all messages from the conversation

        Returns:
            dict: A dictionary in JSON format with an array of chats
        """
        try:
            # Navigate to messages page first
            await self._get_personal_messages()

            # Find all conversation items
            conversation_items = self.page.locator(".ConversationItem-module-root")
            count = await conversation_items.count()

            if count == 0:
                logger.info("No conversation items found")
                return {"chats": []}

            logger.debug(f"Found {count} conversations, will read messages from each")
            all_chats = []

            # Process each conversation
            prev_convo = None
            for i in range(count):
                logger.debug(f"Entering conversation {i}")
                # Get the conversation item
                await self.page.wait_for_timeout(DELAY)
                conversation = conversation_items.nth(i)

                # Get the title of the conversation (product name)
                title_element = conversation.locator(
                    ".hz-Text.hz-Text--bodyLarge"
                ).first

                title = "Unknown conversation"
                if await title_element.count() > 0:
                    title = await title_element.inner_text()
                    logger.info(f"Conversation {i+1}/{count}: {title}")

                # Click on the conversation to open it
                await conversation.click()
                await self.page.wait_for_timeout(DELAY)

                # Handle any modal dialog
                await self._handle_modal_dialog()

                # Parse messages from the conversation
                messages = []

                # Wait for messages to load
                message_groups = self.page.locator(".Messages-module-group")

                # Check if messages are loaded
                max_attempts = 10
                for attempt in range(max_attempts):
                    # Check if message groups are visible
                    if await message_groups.count() > 0:
                        group_count = await message_groups.count()
                        logger.debug(
                            f"Found {group_count} message groups in this conversation"
                        )

                        # Process each message group
                        for group_idx in range(group_count):
                            message_group = message_groups.nth(group_idx)

                            # Extract messages from this group
                            (
                                group_messages,
                                date,
                            ) = await self._extract_messages_from_group(message_group)
                            messages.extend(group_messages)

                            if group_messages:
                                logger.debug(
                                    f"Message group for date: {date} - {len(group_messages)} messages"
                                )

                        await self._handle_modal_dialog()

                        if prev_convo == messages:
                            logger.info(
                                "Previous messages are the same as current ones - implies the weird loading state"
                            )
                            await self.page.wait_for_timeout(5000)
                            messages = []
                            continue
                        prev_convo = messages

                        # If messages were found, break out of the retry loop
                        break
                    else:
                        if attempt < max_attempts - 1:
                            logger.info(
                                f"Messages not appearing (attempt {attempt+1}/{max_attempts}), clicking conversation again..."
                            )

                            while True:
                                await self.page.reload()
                                await self.page.wait_for_timeout(3000)
                                await self._handle_modal_dialog()
                                await self.page.wait_for_timeout(2000)

                                # Find the conversation again and click it
                                conversation_items = self.page.locator(
                                    ".ConversationItem-module-root"
                                )
                                if (await conversation_items.count()) != 0:
                                    break

                            if i < await conversation_items.count():
                                await conversation_items.nth(i).click()
                                await self.page.wait_for_timeout(
                                    2000
                                )  # Wait longer for messages to load
                                await self._handle_modal_dialog()
                        else:
                            logger.warning(
                                "Messages still not appearing after multiple attempts"
                            )

                # Check final state
                if not messages:
                    logger.info("No messages found in this conversation")

                # Clean up the message objects to only include the fields we want in the final output
                clean_messages = []
                for msg in messages:
                    clean_messages.append({"side": msg["side"], "text": msg["text"]})

                # Generate a hash ID for the chat based on its content
                chat_id = self._generate_chat_id(title, messages)

                if chat_id == send_id and send_msg:
                    logger.info(
                        f"Found matching conversation with ID {send_id}, sending message: {send_msg}"
                    )

                    # Find the message input div using the exact selector from the request
                    message_input = self.page.locator(
                        'div.ContentEditable-module-composer.ContentEditable-module-pad.ContentEditable-module-placeholder[contenteditable="true"][spellcheck="true"][data-sem="sendMessageText"][role="textbox"][aria-multiline="true"]'
                    )

                    if await message_input.count() > 0:
                        # Fill the message in the contenteditable div
                        await message_input.press_sequentially(send_msg)

                        # Find and click the send button using the exact selector from the request
                        send_button = self.page.locator(
                            'div.MessageComposer-module-send button.hz-Button.hz-Button--primary[title="Sturen"]'
                        )

                        if await send_button.count() > 0:
                            await send_button.click()
                            logger.info("Send button clicked, message sent")
                            await self.page.wait_for_timeout(
                                2000
                            )  # Wait for message to send
                        else:
                            logger.warning("Could not find the send button")
                    else:
                        logger.warning("Could not find the message input field")

                # Store the conversation data in our new format with ID
                chat_data = {"id": chat_id, "title": title, "messages": clean_messages}
                all_chats.append(chat_data)

                # Re-get the conversation items for the next iteration as the DOM might have changed
                conversation_items = self.page.locator(".ConversationItem-module-root")

            # Return structured JSON format
            return {"chats": all_chats}

        except Exception as e:
            logger.error(f"Error reading conversation messages: {str(e)}")
            return {"chats": []}

    async def send_message(self, id, text):
        await self.read_messages(send_id=id, send_msg=text)

    async def create_post(
        self,
        title: str,
        description: str,
        price: float,
        delivery_option: str = "from home",  # "from home" or "delivery"
        package_size: str = "medium",  # "small", "medium", or "large"
        postcode: str = "1234 AB",  # Postal code in Dutch format
        category: str = None,
        photos: List[str] = None,
        image_data: List[bytes] = None,  # New parameter for image data as bytes
    ) -> str:
        try:
            # Make sure we're on the main page
            await self.page.goto("https://www.marktplaats.nl/")

            # Accept cookies if the dialog appears
            cookie_button = self.page.locator('button[title="Accepteren"]')
            if await cookie_button.count() > 0:
                await cookie_button.click()
                await self.page.wait_for_timeout(DELAY)
                logger.info("Cookie accept clicked")

            # Find and click the "Plaats advertentie" (Place advertisement) button
            # Using the specific URL and attributes
            place_ad_button = self.page.locator(
                'a.hz-Button.hz-Button--primary.hz-Button--small.hz-Button--callToAction[data-role="placeAd"]'
            )

            if await place_ad_button.count() == 0:
                logger.warning(
                    "Could not find the 'Place advertisement' button. Are you logged in?"
                )
                return False, ""

            # Click on the button to go to the ad creation page
            await place_ad_button.click()

            # Check if we need to select between private or business seller
            business_seller_button = self.page.locator(
                'button.hz-Button.hz-Button--primary.hz-Button--full-width:has-text("Zakelijke verkoper")'
            )

            if await business_seller_button.count() > 0:
                logging.info("Found business seller option, clicking it...")
                await business_seller_button.click()

            # Look for the title input field
            title_input = self.page.locator("input#category-keywords")

            if await title_input.count() > 0:
                logging.info(f"Found title input field, entering title: {title}")
                await title_input.fill(title)

                # Click the "Vind categorie" (Find category) button
                find_category_button = self.page.locator("button#find-category")

                if await find_category_button.count() > 0:
                    logging.info("Clicking 'Find category' button...")
                    await find_category_button.click()

                    # Wait a moment for the category to be found
                    await self.page.wait_for_timeout(DELAY)

                    # Click the "Verder" (Continue) button
                    continue_button = self.page.locator(
                        "button#category-selection-submit"
                    )

                    if await continue_button.count() > 0:
                        logging.info("Clicking 'Continue' button...")
                        await continue_button.click()

                        # Determine which image source to use
                        # Priority: 1. image_data (bytes), 2. photos (paths), 3. images folder
                        image_paths = []
                        temp_image_paths = []  # To track any temporary files we create

                        if image_data and len(image_data) > 0:
                            # Use provided image data (bytes)
                            import tempfile

                            for i, img_bytes in enumerate(image_data):
                                # Create a temporary file for each image
                                fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                                with open(fd, "wb") as f:
                                    f.write(img_bytes)
                                temp_image_paths.append(temp_path)

                            image_paths = temp_image_paths
                            logging.info(
                                f"Using {len(image_paths)} images from provided image data"
                            )
                        elif photos and len(photos) > 0:
                            # Use provided photo paths
                            image_paths = photos
                            logging.info(
                                f"Using {len(image_paths)} images from provided paths"
                            )
                        else:
                            logging.error(
                                "No images are provided: I am too scared of getting banned by markplaats to allow this"
                            )
                            return False, ""

                        logging.info(f"Found {len(image_paths)} images to upload")

                        # Find any file input that accepts images (jpg, jpeg, png)
                        # Note: We're using a more general selector to get ALL file inputs
                        for i, image_path in enumerate(image_paths):
                            # Get all file inputs
                            file_inputs = self.page.locator(
                                'input[type="file"][accept*=".jpg"]'
                            )
                            file_inputs_count = await file_inputs.count()

                            if file_inputs_count > 0:
                                # Always upload to the last input element
                                last_input = file_inputs.last
                                logging.info(
                                    f"Uploading image {i+1}/{len(image_paths)}"
                                )
                                await last_input.set_input_files(image_path)

                                # Wait for the upload to complete and for any UI changes
                                await self.page.wait_for_timeout(2000)
                                logging.info(f"Image {i+1}/{len(image_paths)} uploaded")
                            else:
                                logging.info(f"No file inputs found for image {i+1}")
                                break

                        # Clean up temporary files if we created any
                        for temp_path in temp_image_paths:
                            try:
                                os.unlink(temp_path)
                                logging.info(f"Deleted temporary file: {temp_path}")
                            except Exception as e:
                                logging.warning(
                                    f"Failed to delete temporary file {temp_path}: {str(e)}"
                                )

                        # Now fill in the description in the TinyMCE editor
                        # TinyMCE is a bit tricky as it's in an iframe
                        # First, look for the iframe containing the editor
                        description_frame = self.page.frame_locator(
                            'iframe[id*="description"]'
                        )

                        if description_frame:
                            editor_body = description_frame.locator("body#tinymce")

                            if await editor_body.count() > 0:
                                logging.info(
                                    f"Found description editor, entering description: {description}"
                                )
                                # For TinyMCE, we need to use fill or type or html content
                                await editor_body.fill(description)
                                logging.info("Description entered")
                            else:
                                logging.info(
                                    "Could not find the editor body within the iframe"
                                )
                        else:
                            logging.info("Description frame not found")

                        # Now select "Vraagprijs" from the price type dropdown
                        await self.page.wait_for_timeout(1000)
                        price_type_select = self.page.locator(
                            'select[name="price.typeValue"]'
                        )

                        if await price_type_select.count() > 0:
                            logging.info(
                                "Found price type dropdown, selecting 'Vraagprijs'"
                            )
                            # Select "Vraagprijs" which should be value "FREE_BID"
                            await price_type_select.select_option("Vraagprijs")
                            logging.info("Selected 'Vraagprijs'")

                            # Now enter the price
                            price_input = self.page.locator('input[name="price.value"]')

                            if await price_input.count() > 0:
                                logging.info(
                                    f"Found price input field, entering price: {price}"
                                )
                                # Convert float to string with comma as decimal separator
                                price_str = str(price).replace(".", ",")
                                await price_input.fill(price_str)
                                logging.info("Price entered")

                                # Handle the delivery method selection
                                logging.info(f"Delivery option: {delivery_option}")

                                # Find the delivery method radio buttons
                                if delivery_option.lower() == "from home":
                                    # Select "Ophalen" (pick up)
                                    ophalen_radio = self.page.locator(
                                        'input#Ophalen[name="deliveryMethod"]'
                                    )
                                    if await ophalen_radio.count() > 0:
                                        logging.info(
                                            "Selecting 'Ophalen' (pick up) option"
                                        )
                                        await ophalen_radio.check()
                                        logging.info("Selected 'Ophalen'")
                                    else:
                                        logging.info(
                                            "Could not find the 'Ophalen' radio button"
                                        )

                                elif delivery_option.lower() == "delivery":
                                    # Select "Verzenden" (shipping)
                                    verzenden_radio = self.page.locator(
                                        'input#Verzenden[name="deliveryMethod"]'
                                    )
                                    if await verzenden_radio.count() > 0:
                                        logging.info(
                                            "Selecting 'Verzenden' (shipping) option"
                                        )
                                        await verzenden_radio.check()
                                        logging.info("Selected 'Verzenden'")

                                        # Wait for package size options to appear
                                        await self.page.wait_for_timeout(DELAY)

                                        # Now select the package size based on the input
                                        if package_size.lower() == "small":
                                            size_radio = self.page.locator(
                                                'input#S[name="packageSize"]'
                                            )
                                        elif package_size.lower() == "large":
                                            size_radio = self.page.locator(
                                                'input#L[name="packageSize"]'
                                            )
                                        else:  # Default to medium
                                            size_radio = self.page.locator(
                                                'input#M[name="packageSize"]'
                                            )

                                        if await size_radio.count() > 0:
                                            logging.info(
                                                f"Selecting '{package_size}' package size"
                                            )
                                            await size_radio.check()
                                            logging.info(
                                                f"Selected '{package_size}' package size"
                                            )
                                        else:
                                            logging.info(
                                                f"Could not find the '{package_size}' package size radio button"
                                            )
                                    else:
                                        logging.info(
                                            "Could not find the 'Verzenden' radio button"
                                        )
                                else:
                                    logging.info(
                                        f"Unknown delivery option: {delivery_option}"
                                    )

                                # Enter the postcode
                                postcode_input = self.page.locator(
                                    'input[name="contactInformation.postCode"]'
                                )

                                if await postcode_input.count() > 0:
                                    logging.info(
                                        f"Found postcode input, entering: {postcode}"
                                    )
                                    await postcode_input.fill(postcode)
                                    logging.info("Postcode entered")
                                else:
                                    logging.info(
                                        "Could not find the postcode input field"
                                    )

                                # Select the "Free" visibility option
                                # First, look for the div with the free visibility option
                                free_option = self.page.locator(
                                    'div[data-action="select-feature"][data-val="FREE"]'
                                )

                                if await free_option.count() > 0:
                                    logging.info(
                                        "Found 'Free' visibility option, clicking it..."
                                    )
                                    await free_option.click()
                                    logging.info("Selected 'Free' visibility option")
                                else:
                                    logging.info(
                                        "Could not find the 'Free' visibility option"
                                    )

                                # Click the "Naar betalen" (To payment) button to submit the form
                                submit_button = self.page.locator(
                                    "a#syi-place-ad-button"
                                )

                                if await submit_button.count() > 0:
                                    logging.info(
                                        "Found 'Naar betalen' button, clicking it..."
                                    )
                                    if FAKE_POSTING:
                                        logging.warning("NOT DOING ANYTHING CUZ DEBUG")
                                        return True, "some random url"
                                    await self.page.wait_for_timeout(DELAY)
                                    await submit_button.click()
                                    logging.info("Ad submitted successfully!")
                                    return True, self.page.url
                                else:
                                    logging.error(
                                        "Could not find the 'Naar betalen' button"
                                    )
                            else:
                                logging.error("Could not find the price input field")
                        else:
                            logging.error("Could not find the price type dropdown")
                    else:
                        logging.error("Could not find the 'Continue' button")
                else:
                    logging.error("Could not find the 'Find category' button")
            else:
                logging.error("Could not find the title input field")

            # Return False if we couldn't complete the process
            return False, ""

        except Exception as e:
            logging.error(f"Error creating post: {str(e)}")
            return False, ""


async def make_post_sequence(automation, image_data=None):
    sample_title = "Schattige Eenhoorn Knuffel met AWS Startups Shirt"
    sample_description = """Te koop: een superzachte en pluizige eenhoorn knuffel in uitstekende staat! Deze bijzondere eenhoorn draagt een donkerblauw "AWS Startups" T-shirt, wat het een perfect verzamelobject maakt voor techliefhebbers of een leuk cadeau voor jong en oud.

Kleur: Lichtroze met glinsterende hoorn
Ogen: Geborduurd, blauw
Kleding: Origineel AWS Startups shirt
Hoogte: ±25 cm
Staat: Zo goed als nieuw, alleen gebruikt als decoratie
Perfect voor op je bureau, in de kinderkamer of als mascotte voor je startup!"""
    sample_price = 0.69

    # Set all the parameters for the post
    delivery_option = "from home"  # Options: "from home" or "delivery"
    package_size = "large"  # Options: "small", "medium", "large"
    postcode = "1043 DR"  # Example Amsterdam postcode

    logger.info("Starting post creation sequence")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load environment variables from .env file in the script directory
    load_dotenv(os.path.join(script_dir, ".env"))

    # Get credentials from environment variables
    username = os.environ.get("MARKTPLAATS_USERNAME")
    password = os.environ.get("MARKTPLAATS_PASSWORD")

    if not username or not password:
        logger.error("Error: Missing credentials in .env file")
        logger.error(
            "Please create a .env file with MARKTPLAATS_USERNAME and MARKTPLAATS_PASSWORD"
        )
        return

    async with MarktplaatsAutomation(headless=HEADLESS) as automation:
        # Login to Marktplaats
        logged_in = await automation.login(username, password)

        if logged_in:
            logger.info("Successfully logged in to Marktplaats")

        # Use the image_data parameter if provided, otherwise pass None
        ret = await automation.create_post(
            title=sample_title,
            description=sample_description,
            price=sample_price,
            delivery_option=delivery_option,
            package_size=package_size,
            postcode=postcode,
            image_data=image_data,  # Pass the image data if provided
        )

        if ret[0]:
            logger.info("Successfully created the advertisement")
        else:
            logger.error("Failed to create the advertisement")
    return ret


async def main():
    """Example usage of the MarktplaatsAutomation class."""
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load environment variables from .env file in the script directory
    load_dotenv(os.path.join(script_dir, ".env"))

    # Get credentials from environment variables
    username = os.environ.get("MARKTPLAATS_USERNAME")
    password = os.environ.get("MARKTPLAATS_PASSWORD")

    if not username or not password:
        logger.error("Error: Missing credentials in .env file")
        logger.error(
            "Please create a .env file with MARKTPLAATS_USERNAME and MARKTPLAATS_PASSWORD"
        )
        return

    async with MarktplaatsAutomation(headless=HEADLESS) as automation:
        # Login to Marktplaats
        logged_in = await automation.login(username, password)

        if logged_in:
            logger.info("Successfully logged in to Marktplaats")

            # Choose which action to perform (uncomment the desired action)

            # 1. Read and parse all messages from conversations
            logger.info("Starting message monitoring loop")
            while True:
                try:
                    chats = (await automation.read_messages())["chats"]
                    logger.info(json.dumps(chats, indent=2, ensure_ascii=False))
                    for chat in chats:
                        if chat["messages"] and chat["messages"][-1]["side"] != "me":
                            resp = f"You just said: {chat['messages'][-1]['text']}"
                            logger.info(f"Sending mirrored message: {resp}")
                            await automation.send_message(chat["id"], resp)
                            logger.info("Message sent successfully")
                except Exception as e:
                    logger.error(f"Error in message loop: {traceback.format_exc()}")
                await asyncio.sleep(5)

            # 2. Create a new post (uncomment if you want to create a post)
            # await make_post_sequence(automation)
        else:
            logger.error("Failed to log in to Marktplaats")

        # Wait before closing the browser to see the results
        await asyncio.sleep(10)


async def api_post_order_with_login(
    title,
    desc,
    price,
    *,
    delivery="at home",
    package_size="medium",
    postcode="1043 DR",
    image_data=None,
):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Load environment variables from .env file in the script directory
    load_dotenv(os.path.join(script_dir, ".env"))

    # Get credentials from environment variables
    username = os.environ.get("MARKTPLAATS_USERNAME")
    password = os.environ.get("MARKTPLAATS_PASSWORD")

    if not username or not password:
        logger.error("Error: Missing credentials in .env file")
        logger.error(
            "Please create a .env file with MARKTPLAATS_USERNAME and MARKTPLAATS_PASSWORD"
        )
        return

    async with MarktplaatsAutomation(headless=HEADLESS) as automation:
        # Login to Marktplaats
        logged_in = await automation.login(username, password)

        if logged_in:
            logger.info("Successfully logged in to Marktplaats")

        # Use the image_data parameter if provided, otherwise pass None
        ret = await automation.create_post(
            title=title,
            description=desc,
            price=price,
            delivery_option=delivery,
            package_size=package_size,
            postcode=postcode,
            image_data=image_data,  # Pass the image data if provided
        )

        if ret[0]:
            logger.info("Successfully created the advertisement")
        else:
            logger.error("Failed to create the advertisement")
    return ret


if __name__ == "__main__":
    asyncio.run(main())
