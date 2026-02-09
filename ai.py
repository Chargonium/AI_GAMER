import math, random, struct, time

ACTIVATIONS = {
    "linear": lambda x: x,
    "sigmoid": lambda x: 1 / (1 + math.exp(-x)),
    "tanh": lambda x: math.tanh(x),
    "relu": lambda x: max(0, x),
}

ACTIVATIONS_MAP = [
    "linear", "sigmoid", "tanh", "relu"
]


class Node:
    def __init__(self, alpha: float = 0.5, activation = "sigmoid"):
        self.state = 0.0
        self.alpha = alpha
        self.activation_string = activation
        self.activation = ACTIVATIONS.get(activation, lambda x: 1 / (1 + math.exp(-x))) # Default to sigmoid

    def input(self, n):
        self.state = (1 - self.alpha) * self.state + self.alpha * n

    def activate(self):
        return self.activation(self.state)


"""node = Node(alpha=0.5)

node.input(0.2)
node.input(0.8)
node.input(0.2)

node.input(200)


print(node.activate())"""


layers = []

random.seed(39284589)

layers.append([Node(random.random(), ACTIVATIONS_MAP[random.randrange(0,4)]) for _ in range(426*240)]) # Input layer
layers.append([Node(random.random(), ACTIVATIONS_MAP[random.randrange(0,4)]) for _ in range(1200)]) # Hidden layer
layers.append([Node(random.random(), ACTIVATIONS_MAP[random.randrange(0,4)]) for _ in range(1000)]) # Hidden layer
layers.append([Node(random.random(), ACTIVATIONS_MAP[random.randrange(0,4)]) for _ in range(800)]) # Hidden layer
layers.append([Node(random.random(), ACTIVATIONS_MAP[random.randrange(0,4)]) for _ in range(24)]) # Output layer

random.seed(39284589)

weights = []

for i in range(len(layers)-1):
    tmp = []
    for _ in range(len(layers[i]) * len(layers[i+1])):
        tmp.append(random.random())
    weights.append(tmp)

for node in layers[0]: # Input values
    node.input(random.random()*256-128)

now = time.time()

for i in range(len(layers)-1):
    for x, node in enumerate(layers[i]):
        for y, node_2 in enumerate(layers[i+1]):
            #global_index = sum(len(layers[j]) * len(layers[j+1]) for j in range(i)) + x * len(layers[i+1]) + y # Idk how this works but it does, Thanks chatGPT!!!
            node_2.input(node.activate() * weights[i][x])

    
print(f"One pass took: {time.time() - now} seconds")

print([node.activate() for node in layers[-1]] )