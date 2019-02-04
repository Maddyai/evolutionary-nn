import random
from nn import NeuralNetwork

def score(nn_obj):
    return nn_obj.loss

class Evolution():
    def __init__(self, parameters):
        self.parameters = parameters

    def generate_population(self, number):
        population = []

        for _ in range(number):
            nn_obj = NeuralNetwork(self.parameters)
            nn_obj.create_random()
            population.append(nn_obj)

        return population

    def generate_offsprings(self, parent1, parent2):
        childrens = []

        for _ in range(2):
            child = {}
            for parameter in self.parameters:
                child[parameter] = random.choice(
                    [parent1.network[parameter], parent2.network[parameter]])

            nn_obj = NeuralNetwork(self.parameters)
            nn_obj.new_model(child)

            # -----------------------------------randomly allow mutattion
            if 0.2 > random.random():
                nn_obj = self.mutate(nn_obj)

            childrens.append(nn_obj)

        return childrens

    def mutate(self, nn_obj):

        #-----------------------------------get a random key and change its value randomly.
        mutation = random.choice(list(self.parameters.keys()))
        nn_obj.network[mutation] = random.choice(self.parameters[mutation])
        return nn_obj

    def evolve(self, population):
        # ------------------------------------------------Get the population ranked 

        nn_scores = [(score(network), network) for network in population]
        best_model = [x[1] for x in sorted(nn_scores, key=lambda x: x[0], reverse=False)]

        # ------------------------------------------------ Survival of the fittest 
        retain_length = int(len(best_model)*0.4)
        survived = best_model[:retain_length]

        #-----------------------------------------------Randomly allow few to survive.

        for nn in best_model[retain_length:]:
            if 0.1 > random.random():
                survived.append(nn)

        babies = []
        survived_number = len(survived)
        population_limit = len(population) - survived_number

        while len(babies) < population_limit:

            # ----------------------------------Get a random parents.
            parent1 = random.randint(0, survived_number-1)
            parent2 = random.randint(0, survived_number-1)

            if parent1 != parent2:
                parent1 = survived[parent1]
                parent2 = survived[parent2]

                # ------------------------------------allow 2 children
                offsprings = self.generate_offsprings(parent1, parent2)

                for nn in offsprings:
                    if len(babies) < population_limit:
                        babies.append(nn)

        survived.extend(babies)
        return survived
