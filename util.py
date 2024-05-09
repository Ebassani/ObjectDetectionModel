def read_labels_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            labels = file.readlines()
            labels = [label.strip() for label in labels]
        return labels
    except FileNotFoundError:
        print("File not found.")
        return []


def indexes_to_labels(indexes, file):
    labels = []
    labels_from_file = read_labels_from_file(file)
    for index in indexes:
        labels.append(labels_from_file[index])

    return labels

