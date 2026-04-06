from moviepy.audio.io.AudioFileClip import AudioFileClip

files = {
    "E:/CTFS_Harmonic_Fingerprinting/data/raw/sitar/yaman.mp3"    : "E:/CTFS_Harmonic_Fingerprinting/data/raw/sitar/yaman.wav",
    "E:/CTFS_Harmonic_Fingerprinting/data/raw/sitar/bhairavi.mp3" : "E:/CTFS_Harmonic_Fingerprinting/data/raw/sitar/bhairavi.wav",
    "E:/CTFS_Harmonic_Fingerprinting/data/raw/veena/bhairavi.mp3" : "E:/CTFS_Harmonic_Fingerprinting/data/raw/veena/bhairavi.wav",
    "E:/CTFS_Harmonic_Fingerprinting/data/raw/veena/desh.mp3"     : "E:/CTFS_Harmonic_Fingerprinting/data/raw/veena/desh.wav",
}

for src, dst in files.items():
    print(f"Converting {src}...")
    AudioFileClip(src).write_audiofile(dst)
    print(f"  Done!")

print("\nAll converted!")