import tkinter
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
import threading
import torch
import json
import re
# --- NEW: Import AutoConfig to inspect model before loading all weights ---
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import queue
import time
import os

# --- NEW: PDF support requires PyMuPDF ---
try:
    import fitz  # PyMuPDF
except ImportError:
    messagebox.showerror("Missing Dependency", "PyMuPDF is not installed. PDF loading will not work.\nPlease run: pip install PyMuPDF")
    fitz = None

# --- NEW: Custom Widget for Code Blocks ---
class CodeBlock(ctk.CTkFrame):
    # --- NEW: Map common language names to file extensions ---
    LANG_EXT_MAP = {
        "python": "py",
        "javascript": "js",
        "html": "html",
        "css": "css",
        "java": "java",
        "c": "c",
        "cpp": "cpp",
        "csharp": "cs",
        "go": "go",
        "ruby": "rb",
        "php": "php",
        "swift": "swift",
        "kotlin": "kt",
        "rust": "rs",
        "sql": "sql",
        "bash": "sh",
        "shell": "sh",
    }
    
    # --- MODIFIED: Added 'app' parameter for direct access to main window ---
    def __init__(self, master, app, language, code, **kwargs):
        super().__init__(master, **kwargs)
        self.language = language.lower().strip() if language else "text"
        self.code = code.strip()
        self.app = app # Direct reference to the ChatbotApp instance

        # Configure layout
        self.grid_columnconfigure(0, weight=1)

        # Header Frame
        header_frame = ctk.CTkFrame(self, fg_color=("gray85", "gray20"), corner_radius=5)
        header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))
        header_frame.grid_columnconfigure(0, weight=1)

        lang_label = ctk.CTkLabel(header_frame, text=self.language, font=ctk.CTkFont(family="Arial", size=12, weight="bold"))
        lang_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        copy_button = ctk.CTkButton(header_frame, text="Copy", width=60, command=self.copy_code)
        copy_button.grid(row=0, column=1, padx=(0, 5), pady=5)
        
        save_button = ctk.CTkButton(header_frame, text="Save", width=60, command=self.save_code)
        save_button.grid(row=0, column=2, padx=(0, 10), pady=5)
        
        # Code Textbox
        code_textbox = ctk.CTkTextbox(self, font=ctk.CTkFont(family="Consolas", size=13), wrap="none")
        code_textbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        code_textbox.insert("1.0", self.code)
        code_textbox.configure(state="disabled")

        # Configure horizontal scrollbar
        scroll = ctk.CTkScrollbar(self, orientation="horizontal", command=code_textbox.xview)
        scroll.grid(row=2, column=0, sticky="ew", padx=5, pady=(0,5))
        code_textbox.configure(xscrollcommand=scroll.set)
        
    def copy_code(self):
        try:
            self.app.clipboard_clear()
            self.app.clipboard_append(self.code)
            # --- FIXED: Correctly call update_status on the app instance ---
            self.app.update_status("Code copied to clipboard!", "green")
        except Exception as e:
            self.app.update_status(f"Error copying: {e}", "red")

    def save_code(self):
        # --- FIXED: Use the language map for correct extension ---
        file_extension = self.LANG_EXT_MAP.get(self.language, self.language if self.language else "txt")
        file_path = filedialog.asksaveasfilename(
            title="Save Code Snippet",
            defaultextension=f".{file_extension}",
            filetypes=[(f"{file_extension.upper()} files", f"*.{file_extension}"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.code)
            # --- FIXED: Correctly call update_status on the app instance ---
            self.app.update_status(f"Code saved to {file_path.split('/')[-1]}", "green")


# --- Main Application ---
class ChatbotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Local LLM Chatbot Pro by LMLK-seal"); self.geometry("1000x1000"); self.minsize(800, 600)
        
        # --- State Variables ---
        self.tokenizer = None; self.model = None; self.model_path = "No model loaded"
        self.chat_history = []; self.last_bot_response = ""
        self.token_queue = queue.Queue(); self.generation_thread = None; self.stop_generation = False
        self.generation_start_time = None
        # --- NEW: State for file context ---
        self.file_context = ""; self.loaded_filename = ""


        # --- Layout Configuration ---
        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(0, weight=1)
        
        self.chat_scroll_frame = ctk.CTkScrollableFrame(self, corner_radius=0)
        self.chat_scroll_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_scroll_frame.grid_columnconfigure(0, weight=1)

        self.settings_frame = ctk.CTkScrollableFrame(self, label_text="Settings") # Made scrollable for more options
        self.settings_frame.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=10)
        self.settings_frame.grid_columnconfigure(0, weight=1)
        
        bottom_frame = ctk.CTkFrame(self, corner_radius=0)
        bottom_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        bottom_frame.grid_columnconfigure(0, weight=1)

        self.user_input = ctk.CTkEntry(bottom_frame, placeholder_text="Type your message...", font=("Arial", 14))
        self.user_input.grid(row=0, column=0, sticky="ew", padx=(10, 5), pady=10)
        self.user_input.bind("<Return>", self.send_message)
        
        self.status_bar = ctk.CTkLabel(bottom_frame, text="Welcome! Please load a model to begin.", anchor="w", text_color="gray")
        self.status_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))

        self.current_response_label = None
        self._create_settings_panel()
        self.after(100, self._process_token_queue)
        self.bind("<Configure>", self._on_resize)

    def _create_settings_panel(self):
        # Load Model Frame
        load_frame = ctk.CTkFrame(self.settings_frame)
        load_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        load_frame.grid_columnconfigure(0, weight=1)
        self.load_button = ctk.CTkButton(load_frame, text="Load Model", command=self.select_model_folder)
        self.load_button.pack(padx=5, pady=5, fill="x")
        self.model_path_label = ctk.CTkLabel(load_frame, text=f"Model: {self.model_path}", wraplength=180)
        self.model_path_label.pack(padx=5, pady=5, fill="x")

        # --- NEW: GPU Offload Settings Frame ---
        offload_frame = ctk.CTkFrame(self.settings_frame)
        offload_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        offload_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(offload_frame, text="GPU Offload", font=ctk.CTkFont(weight="bold")).pack(pady=(5,0))
        ctk.CTkLabel(offload_frame, text="Layers on GPU (-1 for auto):", wraplength=180).pack(padx=10, anchor="w")
        self.gpu_layers_entry = ctk.CTkEntry(offload_frame)
        self.gpu_layers_entry.insert(0, "-1")
        self.gpu_layers_entry.pack(padx=10, pady=(0, 5), fill="x")
        self.total_layers_label = ctk.CTkLabel(offload_frame, text="Total Layers: N/A", text_color="gray")
        self.total_layers_label.pack(padx=10, pady=(0, 5), anchor="w")

        # Controls Frame
        controls_frame = ctk.CTkFrame(self.settings_frame)
        controls_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        controls_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(controls_frame, text="Controls", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.send_button = ctk.CTkButton(controls_frame, text="Send Message", command=self.send_message, state="disabled",
                                             fg_color="#28a745", hover_color="#218838",
                                             text_color="black", text_color_disabled="#2A2A2A")
        self.send_button.pack(pady=5, padx=5, fill="x")
        self.stop_button = ctk.CTkButton(controls_frame, text="Stop Generation", command=self.handle_stop_generation, state="disabled", fg_color="firebrick", hover_color="#B22222")
        self.stop_button.pack(pady=5, padx=5, fill="x")
        self.clear_button = ctk.CTkButton(controls_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_button.pack(pady=5, padx=5, fill="x")
        self.save_chat_button = ctk.CTkButton(controls_frame, text="Save Chat", command=self.save_chat_history)
        self.save_chat_button.pack(pady=5, padx=5, fill="x")
        self.load_chat_button = ctk.CTkButton(controls_frame, text="Load Chat", command=self.load_chat_history)
        self.load_chat_button.pack(pady=5, padx=5, fill="x")
        
        # --- NEW: File Context Frame ---
        context_frame = ctk.CTkFrame(self.settings_frame)
        context_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        context_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(context_frame, text="File Context", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.load_file_button = ctk.CTkButton(context_frame, text="Load File (TXT/PDF)", command=self.load_file_context)
        self.load_file_button.pack(pady=5, padx=5, fill="x")
        self.clear_file_button = ctk.CTkButton(context_frame, text="Clear File", command=self.clear_file_context, state="disabled")
        self.clear_file_button.pack(pady=5, padx=5, fill="x")
        self.file_context_label = ctk.CTkLabel(context_frame, text="No file loaded.", wraplength=180, text_color="gray")
        self.file_context_label.pack(padx=5, pady=5, fill="x")

        # Statistics Frame
        stats_frame = ctk.CTkFrame(self.settings_frame)
        stats_frame.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
        stats_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(stats_frame, text="Statistics", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.tps_label = ctk.CTkLabel(stats_frame, text="Tokens/sec: N/A")
        self.tps_label.pack(padx=10, pady=(0, 5), anchor="w")

        # Parameters Frame
        params_frame = ctk.CTkFrame(self.settings_frame)
        params_frame.grid(row=5, column=0, padx=10, pady=10, sticky="ew")
        params_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(params_frame, text="Generation Parameters", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        ctk.CTkLabel(params_frame, text="Temperature:").pack(padx=10, anchor="w")
        self.temp_slider = ctk.CTkSlider(params_frame, from_=0.1, to=1.5); self.temp_slider.set(0.6)
        self.temp_slider.pack(padx=10, pady=(0, 10), fill="x")
        ctk.CTkLabel(params_frame, text="Top P:").pack(padx=10, anchor="w")
        self.top_p_slider = ctk.CTkSlider(params_frame, from_=0.1, to=1.0); self.top_p_slider.set(0.9)
        self.top_p_slider.pack(padx=10, pady=(0, 10), fill="x")
        ctk.CTkLabel(params_frame, text="Max New Tokens:").pack(padx=10, anchor="w")
        self.max_tokens_entry = ctk.CTkEntry(params_frame); self.max_tokens_entry.insert(0, "1024")
        self.max_tokens_entry.pack(padx=10, pady=(0, 10), fill="x")

        # System Prompt Frame
        sys_prompt_frame = ctk.CTkFrame(self.settings_frame)
        sys_prompt_frame.grid(row=6, column=0, padx=10, pady=10, sticky="ew")
        ctk.CTkLabel(sys_prompt_frame, text="System Prompt", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.system_prompt_entry = ctk.CTkTextbox(sys_prompt_frame, height=80, wrap="word")
        self.system_prompt_entry.insert("1.0", "You are a helpful AI assistant. Format all code snippets in markdown code blocks.")
        self.system_prompt_entry.pack(padx=5, pady=5, fill="x", expand=True)

        # Theme Switch
        self.theme_switch = ctk.CTkSwitch(self.settings_frame, text="Light Mode", command=lambda: ctk.set_appearance_mode("light" if self.theme_switch.get() else "dark"))
        self.theme_switch.grid(row=7, column=0, padx=10, pady=20, sticky="w")

    def _on_resize(self, event=None):
        width = self.chat_scroll_frame.winfo_width() - 40 
        for widget in self.chat_scroll_frame.winfo_children():
            if isinstance(widget, ctk.CTkFrame):
                for child in widget.winfo_children():
                    if isinstance(child, ctk.CTkLabel) and not isinstance(child.master, CodeBlock):
                         child.configure(wraplength=width)

    # --- MODIFIED: Read GPU layer count from UI and pass to thread ---
    def select_model_folder(self):
        folder_path = filedialog.askdirectory(title="Select Hugging Face Model Folder")
        if not folder_path: return
        
        gpu_layers_str = self.gpu_layers_entry.get().strip()
        try:
            gpu_layers = int(gpu_layers_str)
        except ValueError:
            messagebox.showerror("Invalid Input", "GPU Layers must be an integer (e.g., 20 or -1).")
            return
        
        self.update_status("Loading model, this may take a moment...", "gray")
        self.load_button.configure(state="disabled")
        threading.Thread(target=self._load_model_thread, args=(folder_path, gpu_layers)).start()
        
    # --- MODIFIED: Handle custom device_map creation for GPU offload ---
    def _load_model_thread(self, folder_path, n_gpu_layers):
        try:
            # Step 1: Load config first to get model details
            config = AutoConfig.from_pretrained(folder_path)
            total_layers = getattr(config, 'num_hidden_layers', 0)

            if total_layers == 0:
                 self.after(0, self._update_ui_after_load, folder_path, False, "Could not determine the number of layers from model's config.json.", 0)
                 return

            # Step 2: Determine device_map based on user input
            if not torch.cuda.is_available():
                device_map = "cpu"
                if n_gpu_layers > 0:
                    print("Warning: CUDA not available. Loading model on CPU.") # Log to console
            elif n_gpu_layers == -1:
                device_map = "auto"
            elif 0 <= n_gpu_layers <= total_layers:
                # Build custom device map
                # This assumes a Llama/Mistral-like architecture. May need adjustment for others.
                device_map = {'model.embed_tokens': 0} # 0 corresponds to 'cuda:0'
                for i in range(n_gpu_layers):
                    device_map[f'model.layers.{i}'] = 0
                # Offload the final layers to GPU as well for better performance
                device_map['model.norm'] = 0
                device_map['lm_head'] = 0
                # Layers not in the map will be assigned to CPU by accelerate
            else:
                error_msg = f"Invalid layer count. Model has {total_layers} layers, but {n_gpu_layers} were requested."
                self.after(0, self._update_ui_after_load, folder_path, False, error_msg, total_layers)
                return

            # Step 3: Load tokenizer and model with the determined device_map
            self.tokenizer = AutoTokenizer.from_pretrained(folder_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                folder_path, 
                torch_dtype=torch.bfloat16, 
                device_map=device_map
            )
            self.after(0, self._update_ui_after_load, folder_path, True, None, total_layers)
        except Exception as e:
            self.after(0, self._update_ui_after_load, folder_path, False, str(e), 0)

    # --- MODIFIED: Accept total_layers to update the UI ---
    def _update_ui_after_load(self, folder_path, success, error_msg=None, total_layers=0):
        self.load_button.configure(state="normal")
        if success:
            self.model_path = folder_path.replace("\\", "/").split('/')[-1]
            self.model_path_label.configure(text=f"Model: {self.model_path}")
            self.total_layers_label.configure(text=f"Total Layers: {total_layers}", text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"])
            self.update_status("Model loaded successfully! Ready to chat.", "green")
            self.send_button.configure(state="normal")
            self.clear_chat()
        else:
            self.model = self.tokenizer = None
            self.total_layers_label.configure(text=f"Total Layers: N/A", text_color="gray")
            self.update_status("Failed to load model.", "red")
            messagebox.showerror("Model Load Error", f"Could not load the model.\n\nError: {error_msg}")

    def send_message(self, event=None):
        user_text = self.user_input.get().strip()
        if not user_text or self.model is None or not self.send_button.cget('state') == 'normal': return
        self.user_input.delete(0, "end")
        
        # --- NEW: Prepend file context if it exists ---
        final_user_text = user_text
        if self.file_context:
            final_user_text = f"Based on the content of the file '{self.loaded_filename}', please answer the following question.\n\nFILE CONTENT:\n---\n{self.file_context}\n---\n\nQUESTION:\n{user_text}"
            # We display the original user text in the chat for clarity
            self.render_message(f"[Using context from {self.loaded_filename}]\n\n{user_text}", "You")
        else:
            self.render_message(user_text, "You")
            
        self.chat_history.append({"role": "user", "content": final_user_text})
        self.last_bot_response = ""
        
        self.set_generating_state(True)
        self.stop_generation = False
        
        self.generation_start_time = time.time()
        self.tps_label.configure(text="Tokens/sec: Calculating...")

        self.generation_thread = threading.Thread(target=self._generate_response_thread)
        self.generation_thread.start()

    def _generate_response_thread(self):
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        try:
            system_prompt = self.system_prompt_entry.get("1.0", "end-1c").strip()
            messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
            messages.extend(self.chat_history)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
            generation_kwargs = dict(
                **model_inputs, streamer=streamer, max_new_tokens=int(self.max_tokens_entry.get()),
                do_sample=True, temperature=self.temp_slider.get(), top_p=self.top_p_slider.get()
            )
            generation_thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
            generation_thread.start()
            for token in streamer:
                if self.stop_generation: break
                self.token_queue.put(token)
        except Exception as e:
            self.token_queue.put(f"\n<ERROR>: {e}")
        finally:
            self.token_queue.put(None)

    def _process_token_queue(self):
        try:
            while not self.token_queue.empty():
                token = self.token_queue.get_nowait()
                if token is None:
                    self.handle_generation_finished()
                    return
                
                if self.current_response_label is None:
                    self.current_response_label = ctk.CTkLabel(
                        self.chat_scroll_frame, text="", anchor="w", justify="left",
                        font=ctk.CTkFont(size=14))
                    self.current_response_label.grid(row=self.chat_scroll_frame.grid_size()[1], column=0, sticky="ew", padx=10, pady=5)
                
                self.last_bot_response += token
                self.current_response_label.configure(text=self.last_bot_response)
                self._scroll_to_bottom()
        except queue.Empty:
            pass
        finally:
            self.after(100, self._process_token_queue)
    
    def handle_generation_finished(self):
        if self.current_response_label:
            self.current_response_label.destroy()
            self.current_response_label = None

        self.set_generating_state(False)
        full_response = self.last_bot_response.strip()

        if self.generation_start_time and full_response:
            end_time = time.time()
            duration = end_time - self.generation_start_time
            num_tokens = len(self.tokenizer.encode(full_response))
            tps = num_tokens / duration if duration > 0 else 0
            self.tps_label.configure(text=f"Tokens/sec: {tps:.2f}")
        self.generation_start_time = None

        if "<ERROR>" in full_response:
            self.render_message(full_response, "Error")
            self.update_status("An error occurred during generation.", "red")
            return

        if self.stop_generation:
            full_response += "\n[STOPPED BY USER]"
            self.update_status("Generation stopped by user.", "orange")
        else:
            self.update_status("Ready.", "green")
        
        self.chat_history.append({"role": "assistant", "content": self.last_bot_response.strip()})
        self.render_message(full_response, self.model_path)
    
    def render_message(self, text, author):
        message_frame = ctk.CTkFrame(self.chat_scroll_frame, fg_color="transparent")
        message_frame.grid(row=self.chat_scroll_frame.grid_size()[1], column=0, sticky="ew", pady=(0, 10))
        message_frame.grid_columnconfigure(0, weight=1)

        author_label = ctk.CTkLabel(message_frame, text=author, font=ctk.CTkFont(weight="bold"))
        author_label.grid(row=0, column=0, sticky="w", padx=5)

        pattern = r"```(\w*)\n([\s\S]*?)```"
        parts = re.split(pattern, text)
        
        row_num = 1
        for i, part in enumerate(parts):
            if not part.strip(): continue
            if i % 3 == 0:
                label = ctk.CTkLabel(message_frame, text=part.strip(), anchor="w", justify="left")
                label.grid(row=row_num, column=0, sticky="w", padx=5, pady=2)
            elif i % 3 == 1: continue
            elif i % 3 == 2:
                lang = parts[i-1]; code = part
                # --- MODIFIED: Pass the 'app' instance to CodeBlock ---
                code_block = CodeBlock(message_frame, app=self, language=lang, code=code, fg_color=("gray90", "gray17"))
                code_block.grid(row=row_num, column=0, sticky="ew", padx=5, pady=5)
            row_num += 1
        self._on_resize()
        self._scroll_to_bottom()

    def set_generating_state(self, is_generating):
        if is_generating:
            self.update_status("Generating response...", "orange")
            self.send_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.clear_button.configure(state="disabled")
        else:
            self.send_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.clear_button.configure(state="normal")

    def handle_stop_generation(self):
        self.stop_generation = True; self.update_status("Stopping generation...", "orange")

    def clear_chat(self):
        for widget in self.chat_scroll_frame.winfo_children():
            widget.destroy()
        self.chat_history = []; self.last_bot_response = ""
        if hasattr(self, 'tps_label'):
            self.tps_label.configure(text="Tokens/sec: N/A")
        if self.model: self.update_status("Chat cleared. Ready.", "green")
        while not self.token_queue.empty(): self.token_queue.get()
        if self.current_response_label:
            self.current_response_label.destroy(); self.current_response_label = None

    def update_status(self, text, color):
        self.status_bar.configure(text=text, text_color=color)

    def _scroll_to_bottom(self):
        self.update_idletasks()
        self.chat_scroll_frame._parent_canvas.yview_moveto(1.0)
    
    def save_chat_history(self):
        if not self.chat_history: messagebox.showinfo("Empty Chat", "There is no conversation to save."); return
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                save_data = {"chat_history": self.chat_history, "system_prompt": self.system_prompt_entry.get("1.0", "end-1c").strip()}
                with open(file_path, 'w', encoding='utf-8') as f: json.dump(save_data, f, indent=2)
                self.update_status("Chat history saved.", "green")
            except Exception as e: messagebox.showerror("Save Error", f"Failed to save chat history.\n\nError: {e}")

    def load_chat_history(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
                if "chat_history" not in data: raise ValueError("Invalid chat history file format.")
                self.clear_chat()
                self.chat_history = data.get("chat_history", [])
                system_prompt = data.get("system_prompt", "")
                if system_prompt:
                    self.system_prompt_entry.delete("1.0", "end"); self.system_prompt_entry.insert("1.0", system_prompt)
                for message in self.chat_history:
                    role, content = message.get("role"), message.get("content", "")
                    if role == "user": self.render_message(content, "You")
                    elif role == "assistant": self.render_message(content, self.model_path)
                self.update_status("Chat history loaded.", "green")
            except Exception as e: messagebox.showerror("Load Error", f"Failed to load chat history.\n\nError: {e}")

    # --- NEW: Methods for handling file context ---
    def load_file_context(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("PDF files", "*.pdf"), ("All files", "*.*")])
        if not file_path:
            return
            
        self.update_status(f"Loading file: {os.path.basename(file_path)}...", "orange")
        try:
            content = ""
            if file_path.lower().endswith('.pdf'):
                if fitz is None:
                    messagebox.showerror("PDF Error", "PyMuPDF library is not installed. Cannot read PDF files.")
                    self.update_status("Failed to load PDF (PyMuPDF missing).", "red")
                    return
                with fitz.open(file_path) as doc:
                    content = "".join(page.get_text() for page in doc)
            else: # Assume text file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            
            self.file_context = content
            self.loaded_filename = os.path.basename(file_path)
            self.file_context_label.configure(text=f"Loaded: {self.loaded_filename}", text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"])
            self.clear_file_button.configure(state="normal")
            self.update_status(f"File '{self.loaded_filename}' loaded successfully.", "green")

        except Exception as e:
            self.clear_file_context()
            messagebox.showerror("File Load Error", f"Failed to read the file.\n\nError: {e}")
            self.update_status("Failed to load file.", "red")

    def clear_file_context(self):
        self.file_context = ""
        self.loaded_filename = ""
        self.file_context_label.configure(text="No file loaded.", text_color="gray")
        self.clear_file_button.configure(state="disabled")
        self.update_status("File context cleared.", "gray")

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = ChatbotApp()
    app.mainloop()
