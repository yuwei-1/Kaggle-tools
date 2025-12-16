from pyfiglet import Figlet

f = Figlet(font="starwars")
print(f.renderText("KTOOLS"))

from .logger_setup import setup_logger

logger = setup_logger(__name__)
