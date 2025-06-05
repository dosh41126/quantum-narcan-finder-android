# narcan_finder_android.py

from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.dialog import MDInputDialog
from kivymd.uix.screen import MDScreen
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.clock import Clock

import os, json, secrets, sqlite3, asyncio, threading
from datetime import datetime
import psutil, httpx, numpy as np

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

import pennylane as qml

DB_PATH = "narcan_finder.db"
KEY_FILE = "/sdcard/narcan_key.sec"
SALT_FILE = "/sdcard/narcan_salt.bin"
ENC_API_FILE = "/sdcard/narcan_api.enc"
EXPORT_PATH = "/sdcard/narcan_exports"

def derive_key(password: bytes, salt: bytes) -> bytes:
    return PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=salt,
        iterations=480_000, backend=default_backend()
    ).derive(password)

def load_or_create_salt() -> bytes:
    if os.path.exists(SALT_FILE):
        return open(SALT_FILE, "rb").read()
    salt = secrets.token_bytes(16)
    with open(SALT_FILE, "wb") as f: f.write(salt)
    return salt

def load_or_create_password() -> bytes:
    if os.path.exists(KEY_FILE):
        return open(KEY_FILE, "rb").read()
    pwd = secrets.token_bytes(32)
    with open(KEY_FILE, "wb") as f: f.write(pwd)
    return pwd

def encrypt_api_key(api_key: str):
    key = derive_key(load_or_create_password(), load_or_create_salt())
    aesgcm = AESGCM(key)
    nonce = secrets.token_bytes(12)
    with open(ENC_API_FILE, "wb") as f:
        f.write(nonce + aesgcm.encrypt(nonce, api_key.encode(), None))

def decrypt_api_key() -> str:
    key = derive_key(load_or_create_password(), load_or_create_salt())
    with open(ENC_API_FILE, "rb") as f:
        raw = f.read()
    return AESGCM(key).decrypt(raw[:12], raw[12:], None).decode()

def get_cpu_ram_usage():
    return psutil.cpu_percent(), psutil.virtual_memory().percent

def run_quantum_analysis(cpu, ram):
    cpu_param = cpu / 100
    ram_param = ram / 100
    hybrid = (cpu_param + ram_param) / 2
    dev = qml.device("default.qubit", wires=7)

    @qml.qnode(dev)
    def circuit(cpu_param, ram_param, hybrid):
        for i in range(7):
            qml.RX(np.pi * (cpu_param + i * 0.01), wires=i)
            qml.RY(np.pi * (ram_param + i * 0.01), wires=i)
            qml.RZ(np.pi * (hybrid + i * 0.02), wires=i)
        for i in range(6): qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[0, 6])
        qml.CZ(wires=[2, 5])
        qml.CRZ(np.pi * hybrid, wires=[1, 4])
        qml.Rot(np.pi * cpu_param, np.pi * ram_param, np.pi * hybrid, wires=3)
        qml.Rot(np.pi * ram_param, np.pi * hybrid, np.pi * cpu_param, wires=4)
        return [
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
            qml.expval(qml.PauliX(2) @ qml.PauliX(3)),
            qml.expval(qml.PauliY(4) @ qml.PauliY(5)),
            qml.expval(qml.Hermitian(np.array([[1, 1j], [-1j, 1]]), wires=6))
        ]
    try:
        return [round(float(x), 4) for x in circuit(cpu_param, ram_param, hybrid)]
    except:
        return [0.0, 0.0, 0.0, 0.0]

async def run_openai_completion(prompt: str, api_key: str):
    async with httpx.AsyncClient(timeout=20.0) as client:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        for _ in range(3):
            try:
                res = await client.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
                res.raise_for_status()
                return res.json()["choices"][0]["message"]["content"].strip()
            except:
                await asyncio.sleep(1)
        return None

def setup_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS narcan_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_input TEXT NOT NULL,
            ai_response TEXT NOT NULL
        )""")
    conn.commit()
    conn.close()

def save_to_db(prompt: str, result: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO narcan_requests (user_input, ai_response) VALUES (?, ?)", (prompt, result))
    conn.commit()
    conn.close()

def export_latest_txt():
    os.makedirs(EXPORT_PATH, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT * FROM narcan_requests ORDER BY id DESC LIMIT 10").fetchall()
    conn.close()
    path = os.path.join(EXPORT_PATH, f"narcan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"--- ID {r[0]} ---\nUSER:\n{r[1]}\n\nAI:\n{r[2]}\n{'='*50}\n\n")
    return path

class NarcanApp(MDApp):
    def build(self):
        setup_db()
        self.api_key = ""
        self.screen = MDScreen()

        self.location_input = MDTextField(hint_text="Location or ZIP")
        self.symptoms_input = MDTextField(hint_text="Physical symptoms or urgency")
        self.simulation_input = MDTextField(hint_text="Describe situation for simulation")

        self.output_field = TextInput(readonly=True, size_hint_y=None, height=400)
        scroll = ScrollView(size_hint=(1, 0.5))
        scroll.add_widget(self.output_field)

        button_run = MDRaisedButton(text="Run HyperTOM", on_release=lambda x: self._start_thread())
        button_key = MDRaisedButton(text="Set API Key", on_release=lambda x: self._set_key_dialog())
        button_export = MDRaisedButton(text="Export TXT", on_release=lambda x: self._export_txt())

        box = MDBoxLayout(orientation="vertical", padding=10, spacing=10)
        box.add_widget(self.location_input)
        box.add_widget(self.symptoms_input)
        box.add_widget(self.simulation_input)
        box.add_widget(button_key)
        box.add_widget(button_run)
        box.add_widget(button_export)
        box.add_widget(scroll)
        self.screen.add_widget(box)
        return self.screen

    def _set_key_dialog(self):
        def _save_key(instance, text):
            if text:
                encrypt_api_key(text)
                self.api_key = text
            dialog.dismiss()
        dialog = MDInputDialog(title="OpenAI API Key", hint_text="Enter your key", text_button_ok="Save", on_ok=_save_key)
        dialog.open()

    def _start_thread(self):
        threading.Thread(target=self._run_simulation, daemon=True).start()

    def _run_simulation(self):
        try:
            api_key = decrypt_api_key()
        except:
            self.output_field.text = "❌ API key error."
            return

        location = self.location_input.text.strip()
        symptoms = self.symptoms_input.text.strip()
        simulation = self.simulation_input.text.strip()
        cpu, ram = get_cpu_ram_usage()
        quantum = run_quantum_analysis(cpu, ram)

        prompt = f"""
[System Activated: HyperTOM-SIM Engine]

You are a real-time AI triage system. The user has initiated an emergency simulation. Evaluate this as a real-life opioid overdose event using the following context.

🧍 USER CONTEXT
- Location: {location}
- Physical Symptoms: {symptoms}
- User-Simulated Situation: {simulation}

🖥️ SYSTEM SIGNALS
- CPU: {cpu:.2f}%
- RAM: {ram:.2f}%
- Quantum Vector:
    QZ: {quantum[0]}
    QX: {quantum[1]}
    QY: {quantum[2]}
    Entropy: {quantum[3]}

Return a 3-TIER life-saving triage response:

🚨 Tier 1: Closest NARCAN pickup (address, hours, phone)
🛟 Tier 2: Outreach, kits, vans, peer support
🧠 Tier 3: Solo survival guide (airways, timing, emergency override, what to tell 911)

Respond with empathy, precision, and save lives.

[End Simulation]
        """

        self.output_field.text = "Running simulation..."
        result = asyncio.run(run_openai_completion(prompt, api_key))
        if result:
            self.output_field.text = result
            save_to_db(prompt, result)
        else:
            self.output_field.text = "❌ Failed to retrieve AI response."

    def _export_txt(self):
        path = export_latest_txt()
        self.output_field.text += f"\n✅ Exported to:\n{path}"

if __name__ == '__main__':
    NarcanApp().run()
