import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="meta-llama/llama-4-scout-17b-16e-instruct")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Aarthi, an AI Lead, currently looking for a job.  
            You have been given a job description and a list of relevant links to showcase your portfolio.
            Your job is to write an email to the client regarding the job mentioned above describing the capability of Aarthi 
            in fulfilling their needs. Aarthi was working as an AI Lead at a company called "Micron Technology, Inc", Singapore, 
            where she led a team of AI engineers and data scientists to develop innovative AI solutions.
            Aarthi has a strong background in AI and machine learning, with expertise in natural language processing, computer vision, and deep learning.
            Aarthi has a Master's degree in Data Science and AI from the Nanyang Technological University, Singapore,
            and a Bachelor's degree in Electronics and Instrumentation Engineering from the Anna University, Chennai.
            Aarthi has worked on various AI projects, including developing AI models for predictive maintenance, anomaly detection, 
            and image recognition. She has experience in using popular AI frameworks such as TensorFlow, PyTorch, and Hugging Face Transformers.
            Aarthi is proficient in programming languages such as Python, Java, and C++, and has experience in deploying AI models on cloud platforms like AWS and GCP.
            She has a proven track record of delivering high-quality AI solutions on time and within budget.
            Aarthi is a strong communicator and collaborator, with experience in leading cross-functional teams to deliver AI projects.
            She has also published research papers in top-tier AI conferences and journals.
            Write a professional email to the client, highlighting Aarthi's skills and experience, and how she can contribute to their project.
            Also add the most relevant projects from the following links to showcase Aarthi's portfolio: {link_list}
            Remember you are Aarthi, an AI Lead, and you are writing this email to the client.
            At the end, when you conclude saying Best regards, add my contact details: Aarthi Thirumavalavan, Currently located in London, UK. Mobile: +44 1234 567890, Email:aarthi@gmail.om
            Do not provide a preamble. Make it concise and to the point. Make it viewable without scrolling and make it look like a professional email.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))