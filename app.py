
import gradio as gr
from google import genai
from pypdf import PdfReader
import os

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

def analyze_resume(file, target_role):

    if file is None:
        return "Please upload your resume."

    reader = PdfReader(file)
    resume_text = "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

    prompt = f"""
    You are an expert AI career coach.

    Analyze this student's resume.

    Target Job Role: {target_role}

    Resume:
    {resume_text}

    Give the following:

    1. Resume summary
    2. Technical skills identified
    3. Missing skills for the target role
    4. Strengths and weaknesses
    5. Project recommendations
    6. A personalized 30-day learning roadmap
    7. Resume improvement suggestions

    Be practical and student-friendly.
    Do not invent qualifications or experience.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


with gr.Blocks(title="AI Career Twin") as demo:

    gr.Markdown("# 🚀 AI Career Twin")
    gr.Markdown(
        "Your personal AI-powered career development assistant"
    )

    resume = gr.File(
        label="Upload your PDF Resume",
        file_types=[".pdf"],
        type="filepath"
    )

    role = gr.Dropdown(
        choices=[
            "Data Scientist",
            "Data Analyst",
            "Machine Learning Engineer",
            "Python Developer",
            "AI Engineer"
        ],
        label="Choose your target role"
    )

    button = gr.Button("Analyze My Career")

    output = gr.Markdown(label="AI Career Report")

    button.click(
        fn=analyze_resume,
        inputs=[resume, role],
        outputs=output
    )


demo.launch()
