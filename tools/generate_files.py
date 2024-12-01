import os
import sys

def split_markdown_to_files_with_headers(input_file):
    """
    Splits a markdown file into separate files based on chapter headings.
    Each chapter will be saved as a separate markdown file, with the chapter title as a header.
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    chapter_content = []
    chapter_title = None

    # Ensure output directory exists
    output_dir = "chapters"
    os.makedirs(output_dir, exist_ok=True)

    for line in lines:
        if line.startswith("### **Chapter"):
            # Save the previous chapter if it exists
            if chapter_title and chapter_content:
                save_chapter_with_header(output_dir, chapter_title, chapter_content)
                chapter_content = []
            
            # Set the new chapter title
            chapter_title = line.strip().replace("### **", "").replace("**", "")
        
        chapter_content.append(line)
    
    # Save the last chapter
    if chapter_title and chapter_content:
        save_chapter_with_header(output_dir, chapter_title, chapter_content)

def save_chapter_with_header(output_dir, chapter_title, content):
    """
    Saves a chapter to a markdown file, adding the chapter title as a header.
    """
    filename = f"{chapter_title.replace(' ', '_').replace(':', '').replace('/', '')}.md"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as file:
        # Add the chapter title as a Markdown header
        file.write(f"# {chapter_title}\n\n")
        file.writelines(content)
    
    print(f"Saved: {filepath}")


def main():
    """
    Main function to execute the default functionality when the script is run.
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' does not exist.")
        sys.exit(1)
    
    split_markdown_to_files_with_headers(input_file)

if __name__ == "__main__":
    main()