#!/bin/bash
pdoc --output-dir docs app

# Directory to iterate through
DIR="docs"

# Iterate through each file and each file in every subdirectory
find "$DIR" -type f -name "*.html" | while read -r file; do
    echo "Processing file: $file"

    # Calculate the relative path to the icon.png based on the file's depth
    relative_path=$(dirname "$file" | sed 's|[^/][^/]*|..|g')

    # Use sed to replace the title tag content
    sed -i 's|<title>.*</title>|<title>OsireRAG API Documentation</title>|g' "$file"

    # Update the search bar placeholder to include the dynamic path to the icon
    sed -i "s|<input type=\"search\" placeholder=\"Search...\" role=\"searchbox\" aria-label=\"search\"|<div style=\"display: flex; align-items: center;\">\n<img src=\"$relative_path/icon.png\" width=\"64\" height=\"64\"></img>\n<h1>OsireRAG</h1>\n</div>\n<input type=\"search\" placeholder=\"Search...\" role=\"searchbox\" aria-label=\"search\"|g" "$file"

    if [[ "$(basename "$file")" == "app.html" ]]; then
        sed -i "s|app    </h1>|app    </h1>\n<div class=\"docstring\">\n<h1 id=\"what-is-osirerag\">What is OsireRAG?</h1>\n<p>OsireRAG is an MSOE-developed application which provides basic hybrid-RAG services.</p>\n<p>These services include:</p>\n<ul>\n<li>Document Upload</li>\n<li>Chunking</li>\n<li>Embedding</li>\n<li>Tokenization</li>\n<li>Vectorstore Management</li>\n<li>Keyword Model Management</li>\n<li>Document Retrieval</li>\n<li>Reranking</li>\n<li>LLM Summarization</li>\n</ul>\n<h1 id=\"quickstart\">Quickstart</h1>\n<h2 id=\"quickstart-using-osirerag\">Using OsireRAG</h2>\n<ol>\n<li>Navigate to the <a href=\"https://dh-ood.hpc.msoe.edu\" target=\"_blank\">OOD dashboard</a>\n<ul>\n<li>Make sure you have HPC access first. If you need access, contact your system administrator.</li>\n</ul>\n</li>\n<li>On the OOD dashboard, go to Interactive Apps -> OsireRAG</li>\n<li>Select the number of hours you would like it to run, and specify the purpose.</li>\n<li>Hit \"Launch\"</li>\n<li>Wait for roughly 20 seconds until the application menu goes green and says \"Running\"</li>\n<li>Click the \"Connect to OsireRAG\" button</li>\n</ol>\n<p>You are now connected to OsireRAG! Let's now create a knowledge base and query our knowledge base.</p>\n<ol>\n<li>Go to the \"Projects\" tab at the top</li>\n<li>Enter a name for the knowledge base, vectorstore, and keyword model\n<ul>\n<li>Each knowledge base must be a unique name, and the vectorstore and keyword models must be unique for a given vectorstore.</li>\n</ul>\n</li>\n<li>Hit \"Create Knowledge Base\"</li>\n<li>Go to the \"Home\" tab at the top</li>\n<li>Upload your documents on the left in the \"Upload Documents\" section. <br>\nYou can upload the following document types:\n<ul>\n<li>.txt</li>\n<li>.pdf</li>\n<li>Web URL (In the Upload Website textfield)</li>\n</ul>\n</li>\n</ol>\n<p>The uploading process may take some time depending on the number of documents you are uploading, and the length of these documents. \nAn average textbook-length PDF takes roughly 3 minutes at most to process.</p>\n<p>Once your documents have been uploaded, you can query your knowledge base via the chat interface in the center.</p>\n<p>The LLM response will be streamed to the chat. Once the LLM finishes, the retrieved chunks will be displayed in the \"Retrieved Chunks\" section on the right.</p>\n</div>|g" "$file"
        sed -i "s|app    </h1>|OsireRAG</h1>\n<p>Author: Adam Haile<br>Date: 12/3/2024</p>|g" "$file"
    fi

    sed -i "s|<h2>Submodules</h2>|<h2>Contents</h2>\n<ul>\n<li><a href=\"app.html#what-is-osirerag\">What is OsireRAG?</a></li>\n<li><a href=\"app.html#quickstart\">Quickstart</a>\n<ul>\n<li><a href=\"app.html#quickstart-using-osirerag\">Using OsireRAG</a></li>\n</ul>\n</li>\n</ul>\n<h2>Submodules</h2>|g" "$file"

    sed -i 's|<a class="attribution" title="pdoc: Python API documentation generator" href="https://pdoc.dev" target="_blank">|\n|g' "$file"
    sed -i 's|built with <span class="visually-hidden">pdoc</span><img|\n|g' "$file"
    sed -i 's|alt="pdoc logo"|\n|g' "$file"
    sed -i 's|src="data:image/svg+xml,.*%3C/svg%3E"/>|\n|g' "$file"
done
