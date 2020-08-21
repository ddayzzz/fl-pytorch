from trainers.fedbase import FedBase


class FedAvg(FedBase):

    def __init__(self, options, dataset_info, model):
        super(FedAvg, self).__init__(options=options, model=model, dataset_info=dataset_info, append2metric=None)

    def aggregate(self, solns, num_samples):
        return self.aggregate_parameters_weighted(solns, num_samples)

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i}')

            selected_clients = self.select_clients(round_i=round_i, clients_per_rounds=self.clients_per_round)

            solns, num_samples = self.solve_epochs(round_i, clients=selected_clients)


            self.latest_model = self.aggregate(solns, num_samples)
            # eval on test
            if round_i % self.eval_on_test_every_round == 0:
                self.eval_on(round_i=round_i, clients=self.test_clients, client_type='test')

            if round_i % self.eval_on_train_every_round == 0:
                self.eval_on(round_i=round_i, clients=self.train_clients, client_type='train')

            if round_i % self.save_every_round == 0:
                # self.save_model(round_i)
                self.metrics.write()

        self.metrics.write()