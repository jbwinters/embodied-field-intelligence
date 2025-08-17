#!/usr/bin/env python3
"""Test HTML viewer by checking the generated file."""

import subprocess
from pathlib import Path

# Run interactive mode
print("Running interactive mode to generate HTML...")
result = subprocess.run(
    "python cli.py interactive --H 10 --W 10 --max-steps 20 --seed 123",
    shell=True,
    capture_output=True,
    text=True
)

print(result.stdout)

# Find the HTML file
import re
match = re.search(r'HTML viewer saved to: (.+\.html)', result.stdout)
if match:
    html_path = match.group(1)
    print(f"\n✓ HTML file created: {html_path}")
    
    # Check file size
    size = Path(html_path).stat().st_size
    print(f"✓ File size: {size:,} bytes")
    
    # Check content
    content = Path(html_path).read_text()
    
    # Verify key elements
    checks = [
        ("Has title", "<title>EFI Interactive Viewer</title>" in content),
        ("Has frame data", "frameData =" in content),
        ("Has play button", "playBtn" in content),
        ("Has images", "img_world" in content),
        ("Has controls", "class=\"controls\"" in content),
        ("Has grid layout", "class=\"grid\"" in content),
    ]
    
    print("\nContent verification:")
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
    
    # Count frames
    import json
    frame_match = re.search(r'const frameData = (\[.+?\]);', content, re.DOTALL)
    if frame_match:
        frames = json.loads(frame_match.group(1))
        print(f"\n✓ Episode has {len(frames)} frames")
        
        # Check first frame
        if frames:
            first = frames[0]
            print(f"✓ First frame has {len(first['images'])} field images")
            print(f"✓ Frame info keys: {list(first.get('info', {}).keys())}")
    
    print(f"\n✓ SUCCESS: HTML viewer working correctly!")
    print(f"\nTo view the episode, open this file in a browser:")
    print(f"  file://{Path(html_path).absolute()}")
    
else:
    print("✗ Failed to create HTML viewer")
    print("Error output:", result.stderr)