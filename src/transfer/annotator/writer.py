import csv
import os


class CSVWriter():
    def __init__(self, filename: str):
        """Naive CSV writer implementation used as a naive DB."""
        self.filename = filename

    def n_lines(self) -> int:
        """Get number of lines in a writer."""
        if os.path.exists(self.filename):
            return sum(1 for _ in open(self.filename))
        return 0

    def write(self, elems) -> None:
        """Write the list of elements into the given file."""
        with open(self.filename, 'a+', encoding='utf8') as handle:
            writer = csv.writer(
                handle, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n'
            )
            writer.writerow(elems)
