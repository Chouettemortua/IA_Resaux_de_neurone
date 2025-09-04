import Core.training.AI_training_Quality as ATQ
import Core.training.AI_training_Trouble as ATT
import UI.app as APP

def run():
    choice = int(input("Choisissez une option :\n1. ATQ\n2. ATT\n3. APP\n"))
    if choice == 1:
        ATQ.run_ATQ()
    elif choice == 2:
        ATT.run_ATT()
    elif choice == 3:
        APP.run_app()
    else:
        print("Choix invalide")
        run()

if __name__ == "__main__":
    run()
