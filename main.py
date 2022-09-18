import os
import sys
from pathlib import Path

from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.button import MDRectangleFlatButton
from kivy.core.window import Window
from kivy.utils import platform

os.environ["VERSION"] = " Ver 0.1.3"

if getattr(sys, "frozen", False):  # bundle mode with PyInstaller
    os.environ["ROOT_DIR"] = sys._MEIPASS
else:
    sys.path.append(os.path.abspath(__file__).split("demos")[0])
    os.environ["ROOT_DIR"] = str(Path(__file__).parent)


if (platform == 'android'):
    # need to request premission from android os
    from android.permissions import request_permissions, Permission
    request_permissions([Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE, 
    Permission.BLUETOOTH, Permission.BLUETOOTH_ADMIN])
    if not os.path.exists("/storage/emulated/0/CCHD_Data"):
        os.makedirs("/storage/emulated/0/CCHD_Data")
    os.environ["DATA_DIR"] = "/storage/emulated/0/CCHD_Data"

else:
    if not os.path.exists('CCHD_Data'):
        os.makedirs('CCHD_Data')
    os.environ["DATA_DIR"] = os.path.join(os.environ["ROOT_DIR"],'CCHD_Data')

if not os.path.exists(os.path.join(os.environ["DATA_DIR"], 'tmp')):
    os.makedirs(os.path.join(os.environ["DATA_DIR"], 'tmp'))

if not os.path.exists(os.path.join(os.environ["DATA_DIR"], 'tmp', 'Recorded')):
    os.makedirs(os.path.join(os.environ["DATA_DIR"], 'tmp', 'Recorded'))

if not os.path.exists(os.path.join(os.environ["DATA_DIR"], 'tmp', 'Preprocessed')):
    os.makedirs(os.path.join(os.environ["DATA_DIR"], 'tmp', 'Preprocessed'))

if not os.path.exists(os.path.join(os.environ["DATA_DIR"], 'Tested')):
    os.makedirs(os.path.join(os.environ["DATA_DIR"], 'Tested'))
    

Window.softinput_mode = "below_target"
class MainApp(MDApp):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.theme_cls.material_style = "M2"
        self.theme_cls.primary_palette = "Cyan"
        self.theme_cls.theme_style == "Dark"

    
    def build(self):
        Builder.load_file(
            os.path.join(
                os.environ["ROOT_DIR"], "views", "start_screen", "start_screen.kv"
            )
        )
        Builder.load_file(
            os.path.join(
                os.environ["ROOT_DIR"], "views", "CCHD_test_screen", "CCHD_screen.kv"
            )
        )

        return Builder.load_file(
            os.path.join(
                os.environ["ROOT_DIR"], "views", "screen_manager", "screen_manager.kv"
            )
        )


MainApp().run()