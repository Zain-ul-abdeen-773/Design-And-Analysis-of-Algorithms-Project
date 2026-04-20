import PyPDF2
with open("d:/Study/Design and Analysis of Algorithms/Semester project/Semester_Project_Description.pdf", "rb") as f:
    reader = PyPDF2.PdfReader(f)
    text = []
    for page in reader.pages:
        text.append(page.extract_text())
    
with open("d:/Study/Design and Analysis of Algorithms/Semester project/pdf_content.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(text))