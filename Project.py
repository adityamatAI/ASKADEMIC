from playwright.async_api import async_playwright
import asyncio
import csv
import os

class CUDScraper:
    def __init__(self, username, password, semester, csv_filename="course_offerings.csv"):
        # Initialize login credentials, semester value, and target portal URL
        self.username = username
        self.password = password
        self.semester = semester
        self.url = "https://cudportal.cud.ac.ae/student/login.asp"
        self.playwright = None
        self.browser = None
        self.page = None
        self.csv_filename = csv_filename

    async def start_browser(self, headless=True):
        # Starts a Playwright Chromium browser instance asynchronously
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=headless)
        self.page = await self.browser.new_page()
        print("Browser started.")

    async def login(self):
        try:
            print("Loading login page...")
            await self.page.goto(self.url, timeout=15000)  # 15s timeout

            # Fill in username and password
            await self.page.fill("#txtUsername", self.username)
            await self.page.fill("#txtPassword", self.password)

            # Wait for the term dropdown to appear and then select the term.
            # Update the selector if necessary. 
            await self.page.wait_for_selector("#idterm", timeout=10000)
            await self.page.select_option("#idterm", self.semester)

            # Click the login button
            await self.page.click("#btnLogin")
            await self.page.wait_for_timeout(5000)  # brief pause

            # Wait for the Course Offering link to confirm successful login
            await self.page.wait_for_selector("a:has-text('Course Offering')", timeout=20000)
            print("Login successful.")
        except Exception as e:
            raise RuntimeError(f" Login failed: {e}")



    async def navigate_to_courses(self):
        """Navigates to the Course Offerings page."""
        await self.page.click("text=Course Offering")
        await self.page.wait_for_selector("#displayFilter")
        print(" Navigated to Course Offerings.")

    async def apply_filters(self, division="SEAST"):
        # Applies division filter (e.g., SEAST) to show relevant courses
        await self.page.click("#displayFilter")
        await self.page.select_option("#idDivisions", division)
        await self.page.click("#btnSubmit")
        await self.page.wait_for_selector(".Portal_Group_Table")
        print(f" Filter applied for division: {division}")

    async def scrape_courses(self, filename="course_offerings.csv"):
        """
        Scrapes all course offering pages, parses sessions for each course,
        and saves the structured data into a CSV.
        """
        print(" Starting course scraping...")

        # Dictionary to group sessions by course_code.
        courses_dict = {}
        skipped_courses = 0
        total_pages = 8  # Fixed based on your project spec

        # Group sessions by course_code and handle pages iteratively
        for page_number in range(1, total_pages + 1):
            # Look for session tables and parse each session into the dict
            print(f"\n Scraping Page {page_number}...")
            rows = await self.page.query_selector_all(".Portal_Group_Table tbody tr")
            i = 0
            while i < len(rows):
                # Treat current row as the course row.
                main_row = rows[i]
                course_cells = await main_row.query_selector_all("td")
                if len(course_cells) < 3:
                    i += 1
                    continue

                course_code = (await course_cells[0].inner_text()) or ""
                course_code = course_code.strip()
                course_name = (await course_cells[1].inner_text()) or ""
                course_name = course_name.strip()
                credits = (await course_cells[2].inner_text()) or ""
                credits = credits.strip()

                # Check if next row exists and contains a nested session table.
                session_row = None
                if i + 1 < len(rows):
                    possible_session = rows[i + 1]
                    nested_table = await possible_session.query_selector("table")
                    if nested_table is not None:
                        session_row = possible_session
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1

                # If there's no session row, skip this course.
                if not session_row:
                    print(f" Skipped {course_code} (no session table)")
                    skipped_courses += 1
                    continue

                all_session_rows = await session_row.query_selector_all("tr")
                session_rows = all_session_rows[1:]  # Skip header row

                # Create or update the dictionary entry for the course.
                if course_code not in courses_dict:
                    courses_dict[course_code] = {
                        "course_name": course_name,
                        "credits": credits,
                        "sessions": []
                    }

                for session in session_rows:
                    cells = await session.query_selector_all("td")
                    if len(cells) < 9:
                        continue

                    instructor   = (await cells[1].inner_text()) or ""
                    room         = (await cells[2].inner_text()) or ""
                    days         = (await cells[3].inner_text()) or ""
                    start_time   = (await cells[5].inner_text()) or ""
                    end_time     = (await cells[6].inner_text()) or ""
                    max_enroll   = (await cells[7].inner_text()) or ""
                    total_enroll = (await cells[8].inner_text()) or ""

                    # Strip values to avoid extra whitespace.
                    instructor = instructor.strip()
                    room = room.strip()
                    days = days.strip()
                    start_time = start_time.strip()
                    end_time = end_time.strip()
                    max_enroll = max_enroll.strip()
                    total_enroll = total_enroll.strip()

                    courses_dict[course_code]["sessions"].append({
                        "instructor": instructor,
                        "room": room,
                        "days": days,
                        "start_time": start_time,
                        "end_time": end_time,
                        "max_enroll": max_enroll,
                        "total_enroll": total_enroll
                    })

            # Go to next page 
            if page_number < total_pages:
                try:
                    await self.page.click(f'text="{page_number + 1}"')
                    await self.page.wait_for_timeout(2000)
                except Exception as e:
                    print(f" Failed to click page {page_number + 1}: {e}")
                    break

        # Save parsed and grouped session data into the CSV
        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "No.", "Course", "Course Name", "Credits",
                "Instructor", "Room", "Days",
                "Start Time", "End Time", "Max Enrollment", "Total Enrollment"
            ])

            course_counter = 1  # Increment once per course
            for code, info in courses_dict.items():
                sessions = info["sessions"]
                first_session = True
                for s in sessions:
                    if first_session:
                        writer.writerow([
                            course_counter,
                            code or "",
                            info["course_name"] or "",
                            info["credits"] or "",
                            s["instructor"] or "",
                            s["room"] or "",
                            s["days"] or "",
                            s["start_time"] or "",
                            s["end_time"] or "",
                            s["max_enroll"] or "",
                            s["total_enroll"] or ""
                        ])
                        first_session = False
                    else:
                        writer.writerow([
                            "", "", "", "",
                            s["instructor"] or "",
                            s["room"] or "",
                            s["days"] or "",
                            s["start_time"] or "",
                            s["end_time"] or "",
                            s["max_enroll"] or "",
                            s["total_enroll"] or ""
                        ])
                course_counter += 1

        print(f"\n Done! Saved grouped course data to '{filename}'")
        print(f" Skipped {skipped_courses} course blocks with no session table")

    async def close_browser(self):
        """Closes the browser session."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        print(" Browser closed.")

    async def run(self, headless=True):
        # Full scraping pipeline: login, apply filters, scrape, and close browser
        try:
            await self.start_browser(headless)
            await self.login()
            await self.navigate_to_courses()
            await self.apply_filters()
            await self.scrape_courses(self.csv_filename)
        finally:
            await self.close_browser()

def check_timing_changes(csv_filename="course_offerings.csv", backup_filename="course_offerings_backup.csv"):
    """
    Checks for timing changes by grouping course lecture rows together.
    For each course lecture group (the group starts with a row having a Course value),
    it compares each session's 'Start Time' and 'End Time' with the corresponding session
    in the backup. If any session's timing has changed or a new session is added,
    it outputs a message with the course lecture and the new timings.
    """
    changes = []
    current_rows = []
    # Load current CSV data.
    if os.path.exists(csv_filename):
        with open(csv_filename, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                current_rows.append(row)
    else:
        return ["Current course data not found."]
    
    # Group current data by course lecture.
    current_groups = {}
    group_key = None
    for row in current_rows:
        if row["Course"].strip():
            group_key = row["Course"].strip()
            current_groups[group_key] = [row]
        else:
            if group_key:
                current_groups[group_key].append(row)
    
    # Load backup CSV data and group similarly.
    backup_groups = {}
    if os.path.exists(backup_filename):
        backup_rows = []
        with open(backup_filename, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                backup_rows.append(row)
        group_key = None
        for row in backup_rows:
            if row["Course"].strip():
                group_key = row["Course"].strip()
                backup_groups[group_key] = [row]
            else:
                if group_key:
                    backup_groups[group_key].append(row)
    
    # Compare each course group.
    for course, sessions in current_groups.items():
        if course in backup_groups:
            backup_sessions = backup_groups[course]
            # Compare session by session (using index).
            for idx, cur in enumerate(sessions):
                # If a session row has timing info (non-empty 'Start Time' or 'End Time')
                cur_start = cur["Start Time"].strip() if cur["Start Time"] else ""
                cur_end = cur["End Time"].strip() if cur["End Time"] else ""
                if idx < len(backup_sessions):
                    back = backup_sessions[idx]
                    back_start = back["Start Time"].strip() if back["Start Time"] else ""
                    back_end = back["End Time"].strip() if back["End Time"] else ""
                    if cur_start != back_start or cur_end != back_end:
                        change_msg = (f"Course {course} session {idx + 1} timing changed: "
                                      f"new Start Time: {cur_start}, new End Time: {cur_end}.")
                        changes.append(change_msg)
                else:
                    # New session added
                    change_msg = (f"Course {course} session {idx + 1} is new with timings: "
                                  f"Start Time: {cur_start}, End Time: {cur_end}.")
                    changes.append(change_msg)
    
    # Update the backup CSV with current data (overwrite).
    if current_rows:
        with open(backup_filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=current_rows[0].keys())
            writer.writeheader()
            writer.writerows(current_rows)
    
    return changes

if __name__ == "__main__":
    async def main():
        username = "your_username"
        password = "your_password"
        scraper = CUDScraper(username, password)
        await scraper.run(headless=False)  # Run in non-headless mode for debugging
    asyncio.run(main()) 
