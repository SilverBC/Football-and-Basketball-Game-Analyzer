import cv2
from trackers import Tracker
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
from customtkinter import CTkImage

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True: 
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def check_sports_type(video_frames, tracker):
    tracks = tracker.get_object_tracks(video_frames, stub_path='stubs/track_stubs.pkl')
    rim_detected = any(frame_tracks for frame_tracks in tracks.get("rim", []) if frame_tracks)
    
    if rim_detected:
        court_type = "basketball"
        return tracker, tracks, court_type
    else:
        court_type = "football"
        tracker = tracker = Tracker('train\\runs\\detect\\train\\weights\\best.pt')
        tracks = tracker.get_object_tracks(video_frames, stub_path='stubs/track_stubs.pkl')
        return tracker, tracks, court_type
    
    

def save_video(output_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release
    

def select_video_file(main):
    # Initialize the custom tkinter application
    root = ctk.CTk()
    root.title("Select Video File")
    root.geometry("400x200")

    def open_file_dialog():
        video_file = filedialog.askopenfilename(
            title="Select video file", 
            filetypes=[("MP4 files", "*.mp4"), ("AVI files", "*.avi")]
        )
                 
        root.destroy()  # Close the dialog after selection
        if not video_file:
            video_file = 'input_video\\basketball.mp4'
        
        original_frames, processed_frames = main(video_file)
        display_videos(original_frames, processed_frames)  # Display the videos side by side


    # Create a custom button
    button = ctk.CTkButton(
        master=root, 
        text="Select Video", 
        command=open_file_dialog
    )
    button.place(relx=0.5, rely=0.5, anchor=ctk.CENTER)

    root.mainloop()
    
    
def display_videos(original_frames, processed_frames):
    root = ctk.CTk()
    root.title("Original and Processed Videos")
    root.geometry("1200x600")

    video_height, video_width, _ = original_frames[0].shape
    scale_factor = min(600 / video_height, 600 / video_width)
    resized_width = int(video_width * scale_factor)
    resized_height = int(video_height * scale_factor)

    def resize_frame(frame):
        return cv2.resize(frame, (resized_width, resized_height))

    original_frames = [resize_frame(frame) for frame in original_frames]
    processed_frames = [resize_frame(frame) for frame in processed_frames]

    def update_frames(frame_idx):
        if frame_idx < len(original_frames):
            original_frame = original_frames[frame_idx]
            processed_frame = processed_frames[frame_idx]

            original_image = Image.fromarray(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
            processed_image = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

            original_ctk_image = CTkImage(light_image=original_image, size=(resized_width, resized_height))
            processed_ctk_image = CTkImage(light_image=processed_image, size=(resized_width, resized_height))

            original_label.configure(image=original_ctk_image)
            processed_label.configure(image=processed_ctk_image)
            original_label.image = original_ctk_image
            processed_label.image = processed_ctk_image

            root.after(30, update_frames, frame_idx + 1)

    original_label = ctk.CTkLabel(master=root, text="Original Video")
    processed_label = ctk.CTkLabel(master=root, text="Processed Video")

    original_label.grid(row=0, column=0, padx=10, pady=10)
    processed_label.grid(row=0, column=1, padx=10, pady=10)

    update_frames(0)
    root.mainloop()
    