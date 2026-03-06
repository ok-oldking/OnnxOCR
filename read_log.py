import io
import sys
with io.open('out.txt', 'r', encoding='utf-16le', errors='replace') as f:
    content = f.read()
    # Write to a UTF-8 file instead of printing
    with io.open('out_utf8.txt', 'w', encoding='utf-8') as f_out:
        f_out.write(content)
