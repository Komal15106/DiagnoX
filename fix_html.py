 """
fix_html.py - Remove the orphaned old form-card remnants from index.html
and inject the missing CSS for .tab-btn, .tool-card, .two-col-form, .result-panel
"""
import re

path = r'c:\Users\HP\OneDrive\Desktop\BBY\index.html'
with open(path, encoding='utf-8') as f:
    content = f.read()

# 1. Remove the orphaned block between </section> and the wave SVG that contains old form-card HTML
# It starts right after our new </section> (after result-heart div) and ends before the good wave SVG
# We'll remove the block containing "form-card reveal d1/d2/d3" and the junk wrapping divs/sections
old_block_pattern = r'\n\n  <!-- Symptom Checker -->.*?</section>\n\n  (<svg viewBox="0 0 1440 50")'
content = re.sub(old_block_pattern, r'\n\n  \1', content, flags=re.DOTALL)

# 2. Inject missing CSS before the @media responsive block
new_css = """
    /* ── TAB INTERFACE ── */
    .tab-btn{border:1.5px solid var(--border);border-radius:100px;padding:11px 26px;font-weight:700;font-size:.88rem;cursor:pointer;font-family:inherit;background:#fff;color:var(--muted);transition:all .22s;}
    .tab-btn.active{background:var(--tab-color);color:#fff;border-color:var(--tab-color);box-shadow:0 4px 18px rgba(0,0,0,.15);}
    .tab-btn:hover:not(.active){border-color:var(--tab-color);color:var(--tab-color);}
    .tool-card{background:#fff;border-radius:var(--r);border:1.5px solid var(--border);padding:36px;box-shadow:0 8px 40px rgba(0,0,0,.08);max-width:920px;margin:0 auto;border-top:4px solid var(--card-color);animation:fadeUp .4s ease;}
    .tc-header{display:flex;align-items:center;gap:18px;margin-bottom:28px;padding-bottom:20px;border-bottom:1.5px solid var(--border);}
    .tc-emoji{font-size:2rem;background:var(--card-bg);width:60px;height:60px;border-radius:16px;display:flex;align-items:center;justify-content:center;flex-shrink:0;border:1.5px solid var(--border);}
    .tc-title{font-family:'Syne',sans-serif;font-weight:800;font-size:1.25rem;margin-bottom:4px;}
    .tc-sub{font-size:.8rem;color:var(--muted);}
    .two-col-form{display:grid;grid-template-columns:1fr 1fr;gap:0 32px;}
    @media(max-width:640px){.two-col-form{grid-template-columns:1fr;}}
    .result-panel{margin-top:20px;}

"""

# Insert before the first @media rule
content = content.replace('    @media(max-width:1024px){', new_css + '    @media(max-width:1024px){', 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

lines = content.splitlines()
print(f"Done. Total lines: {len(lines)}")
print("Checking for orphaned form-card:", 'form-card reveal d1' in content)
print("tab-btn CSS present:", '.tab-btn' in content)
print("tool-card CSS present:", '.tool-card' in content)
print("sym-chips present:", 'sym-chips' in content)
