import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.font import Font
#from PIL import Image, ImageTk
import random

# Genetic Algorithm Parameters
POPULATION_SIZE = 10
MUTATION_RATE = 0.2
MAX_GENERATIONS = 1000

# Genetic Algorithm Functions
def create_chromosome():
    return [random.randint(-10, 10) for _ in range(3)]

def fitness(chromosome, target_value, coefficients, sign):
    x, y, z = chromosome
    result = coefficients[0] * x
    result += coefficients[1] * y if sign[0] == '+' else -coefficients[1] * y
    result += coefficients[2] * z if sign[1] == '+' else -coefficients[2] * z
    return -abs(target_value - result)

def select(population, target_value, coefficients, sign):
    population.sort(key=lambda chromosome: fitness(chromosome, target_value, coefficients, sign), reverse=True)
    return population[:2]

def crossover(parent1, parent2):
    point = random.randint(1, 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, 2)
        chromosome[index] += random.choice([-1, 1])
        chromosome[index] = max(-10, min(10, chromosome[index]))
    return chromosome

def genetic_algorithm(target_value, coefficients, sign):
    population = [create_chromosome() for _ in range(POPULATION_SIZE)]
    generation = 0

    while generation < MAX_GENERATIONS:
        selected = select(population, target_value, coefficients, sign)
        best_chromosome = max(population, key=lambda chromo: fitness(chromo, target_value, coefficients, sign))
        best_fitness = fitness(best_chromosome, target_value, coefficients, sign)

        if best_fitness == 0:
            break

        new_population = selected[:]
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population[:POPULATION_SIZE]
        generation += 1

    return best_chromosome, best_fitness, generation

# GUI Functions

def run_algorithm():
    try:
        target_value = int(target_entry.get())
        coefficients = [
            int(coeff_x_entry.get()),
            int(coeff_y_entry.get()),
            int(coeff_z_entry.get())
        ]
        sign = [sign_xy.get(), sign_yz.get()]

        best_solution, best_fitness, generation = genetic_algorithm(target_value, coefficients, sign)

        equation = f"The equation is: {coefficients[0]}x {sign[0]} {coefficients[1]}y {sign[1]} {coefficients[2]}z = {target_value}"

        # Compute the result
        result = coefficients[0] * best_solution[0]
        result += coefficients[1] * best_solution[1] if sign[0] == '+' else -coefficients[1] * best_solution[1]
        result += coefficients[2] * best_solution[2] if sign[1] == '+' else -coefficients[2] * best_solution[2]

        result_text.set(f"{equation}\n"
                        f"Best Solution:\n\nx = {best_solution[0]}\ny = {best_solution[1]}\nz = {best_solution[2]}\n"
                        f"Result: {result}\nGenerations: {generation + 1}\nFitness: {best_fitness}\n")
        
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric inputs.")

# GUI Setup
root = tk.Tk()
root.title("Genetic Algorithm Solver")
root.geometry("600x500")
#
bg_label = tk.Label(root, bg="#d9e6f2")  # Light math-themed blue
bg_label.place(relwidth=1, relheight=1)


# Fonts
title_font = Font(family="Times New Roman", size=20, weight="bold", slant="italic")
input_font = Font(family="Times New Roman", size=14)
output_font = Font(family="Courier New", size=14)

# Title
title_label = tk.Label(root, text="Genetic Algorithm Solver", font=title_font, bg="#ffffff", fg="#000080")
title_label.pack(pady=10)

# Input Frame
input_frame = tk.Frame(root, bg="#f0f0f0", bd=5)
input_frame.place(relx=0.5, rely=0.4, anchor="center", relwidth=0.9, height=200)

# Input Fields
tk.Label(input_frame, text="Target Value:", font=input_font, bg="#f0f0f0").grid(row=0, column=0, padx=10, pady=5)
target_entry = tk.Entry(input_frame, font=input_font)
target_entry.grid(row=0, column=1, pady=5)

tk.Label(input_frame, text="Coefficient for x:", font=input_font, bg="#f0f0f0").grid(row=1, column=0, padx=10, pady=5)
coeff_x_entry = tk.Entry(input_frame, font=input_font)
coeff_x_entry.grid(row=1, column=1, pady=5)

tk.Label(input_frame, text="Coefficient for y:", font=input_font, bg="#f0f0f0").grid(row=2, column=0, padx=10, pady=5)
coeff_y_entry = tk.Entry(input_frame, font=input_font)
coeff_y_entry.grid(row=2, column=1, pady=5)

tk.Label(input_frame, text="Coefficient for z:", font=input_font, bg="#f0f0f0").grid(row=3, column=0, padx=10, pady=5)
coeff_z_entry = tk.Entry(input_frame, font=input_font)
coeff_z_entry.grid(row=3, column=1, pady=5)

tk.Label(input_frame, text="Sign between x and y:", font=input_font, bg="#f0f0f0").grid(row=0, column=2, padx=10, pady=5)
sign_xy = ttk.Combobox(input_frame, values=["+", "-"], font=input_font, width=5)
sign_xy.grid(row=0, column=3, pady=5)

tk.Label(input_frame, text="Sign between y and z:", font=input_font, bg="#f0f0f0").grid(row=1, column=2, padx=10, pady=5)
sign_yz = ttk.Combobox(input_frame, values=["+", "-"], font=input_font, width=5)
sign_yz.grid(row=1, column=3, pady=5)

# Run Button
run_button = tk.Button(root, text="Run Algorithm", font=input_font, bg="#008000", fg="#ffffff", command=run_algorithm)
run_button.pack(pady=10)

# Output
# Output
result_text = tk.StringVar(value="Results will appear here.")
result_label = tk.Label(
    root,
    textvariable=result_text,
    font=output_font,
    bg="#f0f0f0",
    justify="left",
    anchor="nw",
    bd=5,
    relief="sunken",
    wraplength=550  # Allow wrapping if the text is too long
)
result_label.place(relx=0.5, rely=0.85, anchor="center", relwidth=0.9, height=250)  # Increased height to 150


# Run the Tkinter loop
root.mainloop()
