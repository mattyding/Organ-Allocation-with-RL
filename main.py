import random
import numpy as np
from scipy.stats import skewnorm


class Person:
    """
    match_vec: embedding of organ characteristics (organs are similar if their vectors are close together)
    condition_vec:
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
    Class contains the for running simulation.
    """

    def __init__(self):
        self.donor_pool = None
        self.recipient_pool = None
        self.nonce = 0

    def generate_donor(self):
        id = self.nonce
        self.nonce += 1
        age = int(
            min(
                max(
                    20,
                    np.random.normal(50, 10),  # mean age is 50  # sd is 10
                ),
                80,
            )
        )

        race = random.randint(0, 4)  # generates race from 0 to 3

        expiration = int(
            max(
                1,
                np.random.normal(
                    5,  # average donor organ survival length is 5 units of time
                    1,  # down to 2 or up to 8 ish
                ),
            )
        )

        match_vector = np.random.random(3)
        match_vector = match_vector / np.linalg.norm(match_vector)

        condition_vector = random.randint(0, 4)
        return Person(
            id, age, race, expiration, match_vector, np.array([condition_vector]), True
        )

    def generate_recipient(self):
        id = self.nonce
        self.nonce += 1
        age = int(
            min(
                max(20, -np.random.gamma(10, 1) + 80),  # idk wtf these numbers do
                80,
            )
        )

        race = random.randint(0, 4)  # generates race from 0 to 3

        expiration = int(
            max(
                1,
                np.random.normal(
                    5,  # average donor organ survival length is 5 units of time
                    1,  # down to 2 or up to 8 ish
                ),
            )
        )

        match_vector = np.random.random(3)
        match_vector = match_vector / np.linalg.norm(match_vector)

        condition_vector = random.randint(0, 4)
        return Person(
            id, age, race, expiration, match_vector, np.array([condition_vector]), False
        )

    def decrement_expiration(self, verbose):
        score = 0
        for s in [self.donor_pool, self.recipient_pool]:
            ppl_to_remove = []
            for person in s:
                person.expiry -= 1
                if person.expiry == 0:
                    ppl_to_remove.append(person)
                    score += person.age - max(80, person.age)
                    if verbose:
                        print(f"Person {person.id} died at age {person.age}")
            for person in ppl_to_remove:
                s.remove(person)
        return score

    def score(self, donor, recipient):
        # p(d, r) = 1[d_c=r_c] * (d_m * r_m) / (||d_m|| * ||r_m||)
        prob_succ = (donor.condition_vec == recipient.condition_vec) * (
            np.dot(donor.match_vec, recipient.match_vec)
            / np.linalg.norm(donor.match_vec)
            * np.linalg.norm(recipient.match_vec)
        )
        return 0

    def run_simulation(self, max_timesteps, verbose=False):
        """
        Runs one iteration of the simulation.
        """
        self.donor_pool = {
            self.generate_donor() for _ in range(np.random.geometric(1 / 100))
        }
        self.recipient_pool = {
            self.generate_recipient() for _ in range(np.random.geometric(1 / 200))
        }
        if verbose:
            print("Starting simulation...")
            print(f"Donor pool initalized with {len(self.donor_pool)} donors")
            print(
                f"Recipient pool initalized with {len(self.recipient_pool)} recipients"
            )
        rewards = []
        agent = Agent()

        t = 0
        while t < max_timesteps:
            action = agent.get_action()
            if action == (-1, -1):  # advance one timestep
                t += 1
                rewards.append(self.decrement_expiration(verbose))
                if verbose:
                    print("Advancing one timestep...")
            else:
                donor, recipient = action
                if donor in self.donor_pool and recipient in self.recipient_pool:
                    self.donor_pool.remove(donor)
                    self.recipient_pool.remove(recipient)
                    rewards.append(self.score(donor, recipient))
                    if verbose:
                        print(
                            f"Matched donor {donor.id} with recipient {recipient.id} for reward {self.score(donor, recipient)}"
                        )
                else:
                    print("Invalid action: ", action)

        print("Simulation complete.")
        print("Total reward: ", sum(rewards))
        print("Rewards: ", rewards)


class Agent:
    def __init__(self):
        pass

    def get_action(self):
        return (-1, -1)


def main():
    sim = Simulator()
    sim.run_simulation(500, verbose=True)


if __name__ == "__main__":
    main()
