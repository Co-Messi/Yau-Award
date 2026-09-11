"""Extract the acknowledgements pages from the paper PDF."""

import os
import shutil

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "paper_v9_submission.pdf")
OUTPUT = os.path.join(HERE, "acknowledgements_page.pdf")
COPY_DIR = os.environ.get("YAU_PAPER_COPY_DIR")


def main():
    document = fitz.open(PAPER)
    start = next(
        page_number
        for page_number, page in enumerate(document)
        if page.get_text().strip().startswith("Acknowledgements")
    )

    output_document = fitz.open()
    output_document.insert_pdf(
        document, from_page=start, to_page=document.page_count - 1
    )
    output_document.save(OUTPUT)
    print(
        f"wrote acknowledgements_page.pdf "
        f"({output_document.page_count} pages, from paper p{start + 1})"
    )

    output_document.close()
    document.close()

    if COPY_DIR:
        shutil.copy(OUTPUT, COPY_DIR)
        print(f"copied to {COPY_DIR}")


if __name__ == "__main__":
    main()
