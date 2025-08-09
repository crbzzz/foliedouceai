# main.py

class Paul:
    def __init__(self, nom):
        self.nom = nom

    def manger(self):
        """Fonction qui simule le fait de manger."""
        print(f"{self.nom} mange des pâtes avec du pesto.")

    def parler(self, message):
        """Fonction qui simule un discours."""
        print(f"{self.nom} : {message}")

# Fonction principale pour tester la classe Paul
def main():
    paul = Paul("Paul")
    paul.manger()
    paul.parler("Salut les amis !")

if __name__ == "__main__":
    main()