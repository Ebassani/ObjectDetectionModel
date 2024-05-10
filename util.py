def read_labels(file_path):
    try:
        with open(file_path, 'r') as file:
            labels = file.readlines()
            labels = [label.strip() for label in labels]
        return labels
    except FileNotFoundError:
        print("File not found.")
        return []


def indexes_to_labels(indexes, file_path):
    labels = []
    labels_from_file = read_labels(file_path)
    for index in indexes:
        labels.append(labels_from_file[index])

    return labels


def count_labels(labels):
    label_counts = {}
    for label in labels:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    return [[label, count] for label, count in label_counts.items()]


def list_to_string(lst):
    items = []
    for item in lst:
        items.append(f"{item[1]} {item[0]}")
    return ", ".join(items)
