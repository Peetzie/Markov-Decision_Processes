import os
from tabulate import tabulate
import matplotlib.pyplot as plt


class GENV:
    def __init__(self, chap) -> None:
        self.chap = chap
        self.path = f"res/chap{self.chap}"

    def createResDir(self, images=None):
        if images is None:
            if not os.path.exists(path=self.path):
                os.makedirs(self.path)
        else:
            if not os.path.exists(path=f"{self.path}/images"):
                os.makedirs(f"{self.path}/images")

    def save_img(self, img, name):
        img.savefig(f"{self.path}/images/{name}")

    def save_value_iter(self, value_function, assignment, with_steps=False):
        with open(
            f"{self.path}/{assignment}_value_function.txt", "w"
        ) as value_function_file:
            value_function_file.write(assignment + "\n")

            if with_steps:
                value_functions = value_function
                for k, value_function in enumerate(value_functions):
                    # Convert value_function to a list of lists for tabulate
                    value_function_list = [
                        list(map(str, row)) for row in value_function
                    ]

                    # Generate the table using tabulate
                    table = tabulate(value_function_list, tablefmt="fancy_grid")

                    # Write the table to the file

                    value_function_file.write(f"\nStep {k}\n")
                    value_function_file.write(table)
            else:
                # Convert value_function to a list of lists for tabulate
                value_function_list = [list(map(str, row)) for row in value_function]

                # Generate the table using tabulate
                table = tabulate(value_function_list, tablefmt="fancy_grid")

                # Write the table to the file
                value_function_file.write(table)
