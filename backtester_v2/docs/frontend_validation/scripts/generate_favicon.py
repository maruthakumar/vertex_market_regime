#!/usr/bin/env python3
"""
Favicon Generation Script - Enterprise GPU Backtester UI Validation System
Converts MQ_favicon.jpg to multiple favicon formats and sizes for Next.js integration.

Generated as part of Phase 2: Asset Integration & Logo/Favicon Implementation
SuperClaude v3 Enhanced Backend Integration methodology
"""

import os
import sys
from PIL import Image

def generate_favicons(source_path, output_dir):
    """
    Generate multiple favicon formats and sizes from source image.
    
    Args:
        source_path (str): Path to source MQ_favicon.jpg
        output_dir (str): Output directory for generated favicons
    """
    
    # Favicon sizes to generate
    sizes = [
        (16, 16),   # Standard favicon
        (32, 32),   # Standard favicon
        (48, 48),   # Windows shortcut icon
        (96, 96),   # Android Chrome
        (180, 180), # Apple touch icon
        (192, 192), # Android Chrome
        (512, 512)  # PWA maskable icon
    ]
    
    try:
        # Load source image
        print(f"Loading source image: {source_path}")
        with Image.open(source_path) as img:
            # Convert to RGBA for transparency support
            img = img.convert("RGBA")
            print(f"Source image size: {img.size}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate PNG favicons
            for width, height in sizes:
                # Resize with high-quality resampling
                resized = img.resize((width, height), Image.Resampling.LANCZOS)
                
                # Save as PNG
                png_path = os.path.join(output_dir, f"favicon-{width}x{height}.png")
                resized.save(png_path, "PNG", optimize=True)
                print(f"Generated: {png_path}")
                
                # Generate Apple touch icon (180x180)
                if width == 180:
                    apple_path = os.path.join(output_dir, "apple-touch-icon.png")
                    resized.save(apple_path, "PNG", optimize=True)
                    print(f"Generated: {apple_path}")
            
            # Generate ICO file with multiple sizes
            ico_sizes = [(16, 16), (32, 32), (48, 48)]
            ico_images = []
            
            for width, height in ico_sizes:
                resized = img.resize((width, height), Image.Resampling.LANCZOS)
                ico_images.append(resized)
            
            # Save multi-size ICO file
            ico_path = os.path.join(output_dir, "favicon.ico")
            ico_images[0].save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48)])
            print(f"Generated: {ico_path}")
            
            # Generate main favicon.png (32x32)
            main_favicon = img.resize((32, 32), Image.Resampling.LANCZOS)
            main_path = os.path.join(output_dir, "favicon.png")
            main_favicon.save(main_path, "PNG", optimize=True)
            print(f"Generated: {main_path}")
            
            print("\n✅ Favicon generation completed successfully!")
            print(f"📁 Generated files in: {output_dir}")
            print("\n📋 Generated Files:")
            print("   • favicon.ico (16x16, 32x32, 48x48)")
            print("   • favicon.png (32x32)")
            print("   • apple-touch-icon.png (180x180)")
            print("   • favicon-16x16.png")
            print("   • favicon-32x32.png")
            print("   • favicon-48x48.png")
            print("   • favicon-96x96.png")
            print("   • favicon-180x180.png")
            print("   • favicon-192x192.png")
            print("   • favicon-512x512.png")
            
    except Exception as e:
        print(f"❌ Error generating favicons: {e}")
        sys.exit(1)

def main():
    """Main execution function."""
    
    # Define paths
    source_path = "/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/nextjs-app/public/MQ_favicon.jpg"
    output_dir = "/srv/samba/shared/bt/backtester_stable/worktrees/ui-refactor/ui-centralized/nextjs-app/public"
    
    # Verify source file exists
    if not os.path.exists(source_path):
        print(f"❌ Source file not found: {source_path}")
        sys.exit(1)
    
    print("🎨 Enterprise GPU Backtester - Favicon Generation")
    print("=" * 50)
    print("SuperClaude v3 Enhanced Backend Integration")
    print("Phase 2: Asset Integration & Logo/Favicon Implementation")
    print("=" * 50)
    
    generate_favicons(source_path, output_dir)

if __name__ == "__main__":
    main()