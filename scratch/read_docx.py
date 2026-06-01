import zipfile
import xml.etree.ElementTree as ET
import os

docx_path = r"c:\Users\korsr\PycharmProjects\bpm_prediction\docs\Дисертація Коротенко.docx"
out_path = r"c:\Users\korsr\PycharmProjects\bpm_prediction\scratch\dissertation_text.txt"

print(f"Reading {docx_path}...")
if not os.path.exists(docx_path):
    print("Docx file does not exist!")
    exit(1)

os.makedirs(os.path.dirname(out_path), exist_ok=True)

try:
    with zipfile.ZipFile(docx_path) as z:
        doc_xml = z.read("word/document.xml")
        root = ET.fromstring(doc_xml)
        
        # XML namespace for Word
        ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        paragraphs = []
        for p in root.findall('.//w:p', ns):
            p_text = []
            for r in p.findall('.//w:r', ns):
                t = r.find('.//w:t', ns)
                if t is not None and t.text:
                    p_text.append(t.text)
            text = "".join(p_text).strip()
            if text:
                paragraphs.append(text)
                
        print(f"Found {len(paragraphs)} paragraphs. Writing to {out_path}...")
        with open(out_path, "w", encoding="utf-8") as f:
            for p in paragraphs:
                f.write(p + "\n")
        print("Done!")
except Exception as e:
    print(f"Error: {e}")
