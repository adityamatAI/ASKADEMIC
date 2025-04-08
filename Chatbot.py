import sys
import os
import asyncio
import requests
import streamlit as st
import pandas as pd
import nest_asyncio
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, ValidationError
from Project import CUDScraper, check_timing_changes

# Load environment variables from .env file
load_dotenv()

# Windows-specific async fix for Streamlit or notebook environments
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    nest_asyncio.apply()

# Define a structured model for course data using Pydantic
class Course(BaseModel):
    course_code: str
    course_name: str
    credits: str
    instructor: str
    room: str
    days: str
    start_time: str
    end_time: str
    max_enrollment: str
    total_enrollment: str

def load_courses():
    if os.path.exists("course_offerings.csv"):
        df = pd.read_csv("course_offerings.csv", dtype=str)
        df.fillna("", inplace=True)
        df.set_index("No.", inplace=True)

        courses = []
        for _, row in df.iterrows():
            try:
                # Use Pydantic to validate each course row and convert to Course objects
                course = Course(
                    course_code=row["Course"],
                    course_name=row["Course Name"],
                    credits=row["Credits"],
                    instructor=row["Instructor"],
                    room=row["Room"],
                    days=row["Days"],
                    start_time=row["Start Time"],
                    end_time=row["End Time"],
                    max_enrollment=row["Max Enrollment"],
                    total_enrollment=row["Total Enrollment"]
                )
                courses.append(course)
            except ValidationError as e:
                st.warning(f"Error in course data: {e}")

        return courses
    return []

async def run_scraper(username, password, semester):
    # Instantiate the scraper and run the scraping flow asynchronously
    scraper = CUDScraper(username, password, semester)
    await scraper.run(headless=True)


def query_llm(prompt: str) -> str:
    #Sends a prompt to the local LLM via LM Studio's OpenAI-compatible API.
    url = "http://localhost:1047/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "mistral-7b-instruct-v0.3",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1
    }
    resp = requests.post(url, json=data, headers=headers)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"].strip()
    return "Error communicating with LM Studio."

def query_gemini(prompt: str) -> str:
    """
    Sends a prompt to Google's Gemini model
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not set in .env"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    chat = model.start_chat()
    response = chat.send_message(prompt)
    return response.text

def main():
    # Main Streamlit app layout
    st.markdown(
        "<h1 style='text-align: center; color: red;'>ASKADEMIC</h1>",
        unsafe_allow_html=True
    )

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.page = "login"

    # Login
    if st.session_state.page == "login":
        st.title("Login")

        # User input fields
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        # Login Section
        # Get username, password, and semester selection before proceeding
        semester_options = {
            "FA 2024-25": "71",
            "SP 2024-25": "72",
            "SU 2024-25": "73",
            "FA 2025-26": "75"
     }

        # Select box for semesters
        semester_label = st.selectbox("Select Semester", list(semester_options.keys()))
        semester_value = semester_options[semester_label]  # The value you'll send to Playwright

        # Optional: Save to session_state or pass to your scraper
        if st.button("Login"):
            st.session_state.username = username
            st.session_state.password = password
            st.session_state.semester = semester_value  # Use this in your scraper
            st.session_state.page = "dashboard"  # Or whatever the next step is
            st.rerun()

    # Dashboard 
    elif st.session_state.page == "dashboard":
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.page = "login"
            st.session_state.username = ""
            st.session_state.password = ""
            st.session_state.semester = ""
            st.rerun()

        st.title("Course Dashboard")
        # Load scraped course data or run the scraper
        courses = load_courses()

        st.write(f"### Semester: {st.session_state.semester}")  # Display selected semester

        # If data isn't present, allow the user to scrape it
        if not courses:
            st.warning("No course data available. Please scrape first.")
            if st.button("Scrape Data"):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    run_scraper(st.session_state.username, st.session_state.password, st.session_state.semester)
                )
                st.success("Scraping complete! Reloading data...")
                st.rerun()
        else:
            # Convert the list of Pydantic Course objects to a DataFrame for display
            courses_df = pd.DataFrame([course.model_dump() for course in courses])
            st.dataframe(courses_df)

            # Compare the new data with backup CSV to detect scheduling changes
            if st.button("Check Timing Changes"):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    run_scraper(st.session_state.username, st.session_state.password, st.session_state.semester)
                )
                changes = check_timing_changes()
                if changes:
                    st.warning(" Timing changes detected:")
                    for change in changes:
                        st.write(f"- {change}")
                else:
                    st.success(" No changes in timings.")

            st.write("### Chat with the Course Bot!")
            # Chatbot Section — allows user to choose between Gemini and local LLM
            model_choice = st.selectbox(
                "Select chatbot model",
                ["Local LLM", "Gemini"]
            )
            user_input = st.text_input("Your question:")

            if st.button("Send") and user_input:
                if model_choice == "Local LLM":
                    courses_md = courses_df.to_markdown(index=False)
                    prompt = (
                        f"Based on this course data:\n\n{courses_md}\n\n"
                        f"User: {user_input}"
                    )
                    response = query_llm(prompt)
                else:
                    response = query_gemini(user_input)

                st.write(f" **Bot ({model_choice}):** {response}")

if __name__ == "__main__":
    main()