# Assurez-vous que le fichier est bien ouvert en mode lecture
try:
    with open('exemple.txt', 'r') as f:
        contenu = f.read()
except FileNotFoundError:
    print("Le fichier n'existe pas.")
except IOError:
    print("Une erreur est survenue lors de la lecture du fichier.")

# Vérifiez que le contenu n'est pas vide avant de l'afficher
if contenu:
    print(contenu)
else:
    print("Le fichier est vide.")