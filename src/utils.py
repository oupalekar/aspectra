def map_class_to_integer(class_name):
    match class_name:
        case "STAR":
            return 0
        case "GALAXY":
            return 1
        case "QSO":
            return 2