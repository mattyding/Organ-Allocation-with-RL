import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from scipy.stats import skewnorm


class Person:
    """
    match_vec: embedding of organ characteristics (organs are similar if their vectors are close together)
    condition_vec: embedding of organ condition
    """

    def __init__(self, id, age, race, expiry, match_vec, condition_vec, donor):
        self.id = id
        self.age = age
        self.race = race
        self.expiry = expiry
        self.match_vec = match_vec
        self.condition_vec = condition_vec
        self.donor = donor  # True for donors, False for recipients


class Simulator:
    """
    Contains the code for running a simulation.
    """

    def __init__(self, verbose=False):
        self.donor_pool = None
        self.recipient_pool = None
        self.nonce = 0
        self.verbose = verbose

    def generate_donor(self):
        id = self.nonce
        self.nonce += 1
        age = int(
            min(
                max(
                    20,
                    np.random.normal(
                        50, 10
                    ),  # mean age is 50, standard deviation is 10
                ),
                80,
            )
        )

        race = random.randint(0, 4)  # generates race from 0 to 4 inclusive

        expiration = int(
            max(
                1,
                np.random.normal(
                    5,  # average survival length is 5 units of time
                    1,  # range between around 2 units to 8 units
                ),
            )
        )

        match_vector = np.random.random(3)
        match_vector = match_vector / np.linalg.norm(match_vector)

        condition_vector = 0
        return Person(
            id, age, race, expiration, match_vector, np.array([condition_vector]), True
        )

    def generate_recipient(self):
        id = self.nonce
        self.nonce += 1
        age = int(
            min(
                max(
                    20, -np.random.gamma(10, 1) + 80
                ),  # mean of 70, standard deviation of sqrt(10)
                80,
            )
        )

        race = random.randint(0, 4)  # generates race from 0 to 4 inclusive

        expiration = int(
            max(
                1,  # can't have lower than 1 unit of expiration
                np.random.normal(
                    5,  # average survival length is 5 units of time
                    1,  # range between around 2 units to 8 units
                ),
            )
        )

        match_vector = np.random.random(3)
        match_vector = match_vector / np.linalg.norm(match_vector)

        condition_vector = 0
        return Person(
            id, age, race, expiration, match_vector, np.array([condition_vector]), False
        )

    def decrement_expiration(self):
        score = 0
        for s in [self.donor_pool, self.recipient_pool]:
            ppl_to_remove = []
            for person in s:
                person.expiry -= 1
                if person.expiry == 0:
                    ppl_to_remove.append(person)
                    score += person.age - max(80, person.age)
                    if self.verbose:
                        print(f"Person {person.id} died at age {person.age}")
            for person in ppl_to_remove:
                s.remove(person)
        if self.verbose:
            print(f"Loss from deaths: {score}")
        return score

    def add_new_participants(self):
        for _ in range(np.random.geometric(1 / 5)):
            self.donor_pool.add(self.generate_donor())
        for _ in range(np.random.geometric(1 / 20)):
            self.recipient_pool.add(self.generate_recipient())

    def score(self, donor, recipient):
        # probability of success is computed using cosine similarity
        # p(d, r) = 1[d_c=r_c] * (d_m * r_m) / (||d_m|| * ||r_m||)
        prob_succ = (donor.condition_vec == recipient.condition_vec) * (
            np.dot(donor.match_vec, recipient.match_vec)
            / (np.linalg.norm(donor.match_vec) * np.linalg.norm(recipient.match_vec))
        )
        # fitness score (prob % scaled by mismatch in donors/recipients)
        fitness = (
            prob_succ[0]
            * (1 + len(self.recipient_pool))
            / (1 + len(self.donor_pool))
            * 100
        )
        return fitness

    def run_simulation(self, max_timesteps):
        """
        Runs one iteration of the simulation.
        """
        self.donor_pool = {
            self.generate_donor() for _ in range(np.random.geometric(1 / 50))
        }
        self.recipient_pool = {
            self.generate_recipient() for _ in range(np.random.geometric(1 / 200))
        }
        if self.verbose:
            print("Starting simulation...")
            print(f"Donor pool initalized with {len(self.donor_pool)} donors")
            print(
                f"Recipient pool initalized with {len(self.recipient_pool)} recipients"
            )
        rewards = []
        input_dim = 2 * next(self.recipient_pool.__iter__()).match_vec.shape[0]
        self.agent = MLPAgent(input_dim=input_dim)

        t = 0
        while t < max_timesteps:
            action = self.agent.get_action(self.donor_pool, self.recipient_pool)
            donor, recipient = action
            if action == (-1, -1):  # advance one timestep
                t += 1
                reward = self.decrement_expiration()
                self.add_new_participants()
                if self.verbose:
                    print("Advancing one timestep...")
            else:
                if donor in self.donor_pool and recipient in self.recipient_pool:
                    self.donor_pool.remove(donor)
                    self.recipient_pool.remove(recipient)
                    reward = self.score(donor, recipient)

                    if self.verbose:
                        print(
                            f"Matched donor {donor.id} with recipient {recipient.id} for reward {self.score(donor, recipient)}"
                        )
                else:
                    print("Invalid action: ", action)

            self.agent.update(donor, recipient, reward)
            rewards.append(reward)

        print("Simulation complete.")
        print("Rewards: ", rewards)
        print("Total reward: ", sum(rewards))

    def plot_results(self):
        if not self.agent:
            print("No simulation history recorded.")
            return
        plt.clf()
        plt.plot(self.agent.threshold_history)
        plt.xlabel("Timestep")
        plt.ylabel("Minimum Score Threshold for Matching")
        plt.title("MLPAgent Learned Matching Thresholds")
        plt.savefig("figures/MLPAgent_thresholds.png")


class MLPAgent:
    # reinforcement learning agent that uses a MLP to evaluate pairings
    def __init__(self, input_dim):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.last_prediction = None
        torch.autograd.set_detect_anomaly(True)

        self.threshold = None  # min threshold for accepting a pairing
        self.alpha = 0.01  # learning rate
        self.threshold_history = []

    def get_action(self, donor_pool, recipient_pool):
        # determine the best pairing of all possible pairings
        # returns (-1, -1) if all pairings have negative reward
        # trains itself using the reward
        if len(donor_pool) == 0 or len(recipient_pool) == 0:
            return (-1, -1)

        best_pairing = None
        best_reward = None
        for donor in donor_pool:
            for recipient in recipient_pool:
                reward = self.model(
                    torch.tensor(
                        np.concatenate([donor.match_vec, recipient.match_vec], axis=0),
                        dtype=torch.float,
                    )
                )
                if best_reward is None or reward.item() > best_reward.item():
                    best_reward = reward.clone()
                    best_pairing = (donor, recipient)

        self.last_prediction = best_reward
        if self.threshold is None:
            self.threshold = best_reward.item()
        if best_reward.item() < self.threshold:
            return (-1, -1)
        return best_pairing

    def update(self, donor, recipient, reward):
        if type(donor) == Person:
            # train the model on the last prediction
            self.optimizer.zero_grad()
            loss = self.loss(
                self.last_prediction, torch.tensor([reward], dtype=torch.float)
            )
            loss.backward(retain_graph=True)
            self.optimizer.step()

        # update threshold based on rewards
        # negative rewards (lots of deaths) decrease
        self.threshold = max(self.threshold, self.threshold + self.alpha * reward)
        self.threshold_history.append(self.threshold)


def main():
    sim = Simulator(verbose=True)
    sim.run_simulation(200)
    sim.plot_results()


if __name__ == "__main__":
    main()
