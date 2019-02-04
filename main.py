from evolution import Evolution
import matplotlib.pyplot as plt
import seaborn as sns

def find_average_loss(networks):
    t = 0

    for network in networks:
        t += network.loss

    avg = t / len(networks)
    return avg

if __name__ == '__main__':
    generations, population = 40, 30

    parameters = {
        "no_neurons": [1,2,3,4,5,6,7,8,9,10], # 4,
        "momentum_rate": [0.001 , 0.01, 0.1, 0.2, 0.4, 0.6], #0.01,
        "lambda_rate": [0.001, 0.01, 0.1, 0.2, 0.4, 0.6], #0.001,
        "learning_rate": [0.1, 0.2, 0.4, 0.5, 0.6], #0.5,
    }
    history_of_errors = []
    evolution_obj = Evolution(parameters)
    networks = evolution_obj.generate_population(population)

    for generations_idx in range(1, generations+1):
        print ("Generation - {} / {} :".format(generations_idx, generations))

        for network in networks:
            network.fit()

        average_loss = find_average_loss(networks)
        history_of_errors.append(average_loss)

        print ("Average Loss : {}".format(average_loss))
        print ("----------------------------------------------------")

        if (generations_idx != generations):
            networks = evolution_obj.evolve(networks)

    #----------------------------------------- Sort Final Generation Population.

    networks = sorted(networks, key=lambda x: x.loss, reverse=False)

    #-------------------------------------Top 3 Winners of the Evolutions.
    for network in networks[:3]:
        print (network.network)
        print (network.loss)


    #---------------------------------------for visualize
    plt.plot(list(range(1, generations+1)), history_of_errors)

    # sns.lmplot(x='Attack', y='Defense', data=(range(1, generations+1), history_of_errors))
    # seaborn.scatterplot(x=None, y=None,
    plt.xlabel('No. of Generations') 
    plt.ylabel('Average RMSE')   
    plt.title('History of NN Generations!') 
    plt.show()
