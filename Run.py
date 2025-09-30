import tkinter as tk
from tkinter import *
from tkinter import messagebox as tm
from PIL import Image, ImageTk
import PIL
import cv2
import os
from itertools import count, cycle
import generate_data as gd
import cnn as train
import main as signtotext
import tkinter.ttk as ttk


bgcolor = "#DAF7A6"      
bgcolor1 = "#7DAA3C"     
fgcolor = "#1B3B00"      

# Paths
op_dest = "D:/Projectcode/Signlanguage/filtered_data/"
alpha_dest = "D:/Projectcode/Signlanguage/alphabet/"

# Colors and Fonts
bgcolor = "#F0F8FF"
accent_color = "#4CAF50"
button_color = "#00796B"
text_color = "#333333"
font_main = ('Helvetica', 18, 'bold')
font_sub = ('Helvetica', 14)



# File mapping
file_map = {}
editFiles = [f for f in os.listdir(op_dest) if f.endswith(".webp")]
for i in editFiles:
    file_map[i] = i.replace(".webp", "").split()

class ImageLabel(tk.Label):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self._job = None  # Initialize the job attribute to None

    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)

        frames = []
        try:
            for i in count(0):
                frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i+1)
        except EOFError:
            pass

        self.frames = cycle(frames)
        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100

        if len(frames) == 1:
            self.config(image=next(self.frames))
        else:
            self.next_frame()


    def unload(self):
        if self._job:
            self.after_cancel(self._job)
            self._job = None
        self.config(image=None)
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.config(image=next(self.frames))
            self._job = self.after(self.delay, self.next_frame)

    def update_speed(self, speed):
        self.delay = speed
        if self._job:
            self.after_cancel(self._job)
            self._job = None
        if hasattr(self, 'frames') and self.frames:
            self.next_frame()


def check_sim(i, file_map):
    for item in file_map:
        for word in file_map[item]:
            if i == word:
                return 1, item
    return -1, ""


def func(a, speed=300):
    all_frames = []
    final = PIL.Image.new('RGB', (380, 260))
    words = a.split()
    for i in words:
        flag, sim = check_sim(i, file_map)
        if flag == -1:
            for j in i:
                im = PIL.Image.open(alpha_dest + str(j).lower() + "_small.gif")
                frameCnt = im.n_frames
                for frame_cnt in range(frameCnt):
                    im.seek(frame_cnt)
                    im.save("tmp.png")
                    img = cv2.imread("tmp.png")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (380, 260))
                    im_arr = PIL.Image.fromarray(img)
                    for itr in range(15):
                        all_frames.append(im_arr)
        else:
            im = PIL.Image.open(op_dest + sim)
            im.info.pop('background', None)
            im.save('tmp.gif', 'gif', save_all=True)
            im = PIL.Image.open("tmp.gif")
            frameCnt = im.n_frames
            for frame_cnt in range(frameCnt):
                im.seek(frame_cnt)
                im.save("tmp.png")
                img = cv2.imread("tmp.png")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (380, 260))
                im_arr = PIL.Image.fromarray(img)
                all_frames.append(im_arr)
    final.save("out.gif", save_all=True, append_images=all_frames, duration=speed, loop=0)
    return all_frames


def Home():
    window = tk.Tk()
    window.title("Sign Language Translator")
    window.geometry('1280x720')
    window.configure(background=bgcolor)
    window.resizable(False, False)

    # Setup ttk style
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('Custom.TButton',
                    background=bgcolor1,
                    foreground=fgcolor,
                    font=('Helvetica', 15, 'bold'),
                    padding=8)
    style.map('Custom.TButton',
              background=[('active', '#92C66F')],
              foreground=[('active', 'white')])

    style.configure('Custom.TRadiobutton',
                    background=bgcolor,
                    foreground=fgcolor,
                    font=('Times New Roman', 16, 'bold'),
                    padding=6)
    style.map('Custom.TRadiobutton',
              background=[('active', '#7DAA3C')],
              foreground=[('active', 'white')])

    # Header Label
    header = tk.Label(window,
                      text="Sign Language Translation",
                      font=('Segoe UI', 28, 'bold underline italic'),
                      bg=bgcolor, fg=fgcolor)
    header.place(x=350, y=10)

    # Input Text Label and Entry
    lbl_text = tk.Label(window, text="Enter Your Text", font=('Arial', 18, 'bold'),
                        bg=bgcolor, fg=fgcolor)
    lbl_text.place(x=100, y=100)
    txt = tk.Entry(window, width=25, font=('Arial', 16), bd=3, relief='groove')
    txt.place(x=350, y=105)

    # Caption Label and Entry
    lbl_caption = tk.Label(window, text="Enter Your Caption", font=('Arial', 18, 'bold'),
                           bg=bgcolor, fg=fgcolor)
    lbl_caption.place(x=100, y=250)
    txt1 = tk.Entry(window, width=25, font=('Arial', 16), bd=3, relief='groove')
    txt1.place(x=350, y=255)

    # Radio Buttons for mode selection
    var = tk.IntVar(value=1)
    r1 = ttk.Radiobutton(window, text="Sign2Text", variable=var, value=1, style='Custom.TRadiobutton')
    r1.place(x=350, y=170)
    r2 = ttk.Radiobutton(window, text="Text2Sign", variable=var, value=2, style='Custom.TRadiobutton')
    r2.place(x=470, y=170)

    # Label to show selected mode
    label = tk.Label(window, text="", font=('Arial', 16, 'bold'), bg=bgcolor, fg=fgcolor)
    label.place(x=350, y=210)

    # Animated GIF display label
    lbl_img = ImageLabel(window, bd=5, relief='sunken')
    lbl_img.place(x=350, y=300, width=380, height=260)

    # Speed Slider
    speed_var = tk.IntVar(value=200)
    def on_speed_change(value):
        speed = int(float(value))
        lbl_img.update_speed(speed)

    speed_label = tk.Label(window, text="Animation Speed (ms):", font=('Arial', 14), bg=bgcolor, fg=fgcolor)
    speed_label.place(x=770, y=520)
    speed_slider = ttk.Scale(window, from_=50, to=1000, orient='horizontal',
                             variable=speed_var, command=on_speed_change)
    speed_slider.place(x=770, y=550, width=180)

    # Button Functions
    def clear():
        txt.delete(0, 'end')
        txt1.delete(0, 'end')
        label.config(text="")
        lbl_img.unload()

    def sel():
        selection = var.get()
        label.config(text="Mode: " + ("Sign2Text" if selection == 1 else "Text2Sign"))

    def sign2text():
            mode = var.get()
            if mode == 1:
                # === Sign2Text Functionality ===
                signtotext.process()  # <-- Call your actual function to do Sign2Text
                
            else:
                # === Text2Sign Functionality ===
                text = txt.get().strip()
                if not text:
                    tm.showwarning("Input needed", "Please enter text to generate GIF")
                    return
                try:
                    current_speed = int(float(speed_var.get()))
                    func(text, speed=current_speed)
                    lbl_img.load('out.gif')
                    lbl_img.update_speed(current_speed)
                except Exception as e:
                    tm.showerror("Error", f"Text2Sign failed:\n{e}")


    def datacreation():
                text=txt1.get()
                if text!="":
                        gd.process(text)
                else:
                         tm.showinfo("Error","Enter the Caption Letter")

    def trainprocess():
        tm.showinfo("Info", "Training process started")
        train.process()

    def train_cnn():
        tm.showinfo("Training CNN", "CNN model training started...")
        import cnn
        cnn.process()

    def train_mobilenet():
        tm.showinfo("Training MobileNetV2", "MobileNetV2 model training started...")
        import mobilenet
        mobilenet.process()
        
    def train_dense():
        tm.showinfo("Training DenseNet121", "DenseNet121 model training started...")
        import DenseNet121
        DenseNet121.process()

    
    # Bind radio buttons to selection handler
    var.trace_add('write', lambda *args: sel())

    # Buttons
    start_btn = ttk.Button(window, text="Start", command=sign2text, style='Custom.TButton')
    start_btn.place(x=800, y=105, width=140, height=40)

    clear_btn = ttk.Button(window, text="Clear", command=clear, style='Custom.TButton')
    clear_btn.place(x=350, y=580, width=140, height=40)

    data_btn = ttk.Button(window, text="Data Creation", command=datacreation, style='Custom.TButton')
    data_btn.place(x=800, y=255, width=180, height=40)

    train_btn = ttk.Button(window, text="Train", command=trainprocess, style='Custom.TButton')
    train_btn.place(x=800, y=320, width=140, height=40)
    
    cnn_btn = ttk.Button(window, text="Train Custom CNN", command=train_cnn, style='Custom.TButton')
    cnn_btn.place(x=800, y=320, width=200, height=40)

    mobilenet_btn = ttk.Button(window, text="Train MobileNet", command=train_mobilenet, style='Custom.TButton')
    mobilenet_btn.place(x=800, y=370, width=180, height=40)

    resnet_btn = ttk.Button(window, text="Train DenseNet121", command=train_dense, style='Custom.TButton')
    resnet_btn.place(x=800, y=420, width=200, height=40)

    quit_btn = ttk.Button(window, text="Quit", command=window.destroy, style='Custom.TButton')
    quit_btn.place(x=1100, y=580, width=140, height=40)

    sel() 

    window.mainloop()
# Call Home
Home()

