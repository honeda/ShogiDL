import time  # noqa F401
from concurrent.futures import ThreadPoolExecutor

class BasePlayer():
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None

    def usi(self):
        pass

    def usinewgame(self):
        pass

    def setoption(self, args):
        pass

    def isready(self):
        pass

    def position(self, sfen, usi_moves):
        pass

    def set_limits(self, btime=None, wtime=None, byoyomi=None, binc=None,
                   winc=None, nodes=None, infinite=False, ponder=False):
        """

        Args:
            btime (int, optional): black's time condition. [ms]
                Defaults to None.
            wtime (int, optional): white's time condition. [ms]
                Defaults to None.
            byoyomi (int, optional): byo-yomi time. [ms] Defaults to None.
            binc (int, optional): black's time added per move under
                the Fischer rule. Defaults to None.
            winc (int, optional): white's time added per move under
                the Fischer rule. Defaults to None.
            nodes (_type_, optional): _description_. Defaults to None.
            infinite (bool, optional): if True, no time limit.
                Defaults to Flase.
            ponder (bool, optional): if True, USI_Ponder is ON.
                Defaults to False.
        """
        pass

    def go(self):
        pass

    def stop(self):
        pass

    def ponderhit(self, last_limits):
        pass

    def quit(self):
        pass

    def run(self):
        while True:
            cmd_line = input().strip()
            cmd = cmd_line.split(" ", 1)

            if cmd[0] == "usi":
                self.usi()
                print("usiok", flush=True)
            elif cmd[0] == "setoption":
                option = cmd[1].split(" ")
                self.setoption(option)
            elif cmd[0] == "isready":
                self.isready()
                print("readyok", flush=True)
            elif cmd[0] == "usinewgame":
                self.usinewgame()
            elif cmd[0] == "position":
                args = cmd[1].split("moves")
                self.position(args[0].strip(), args[1].split() if len(args) > 1 else [])
            elif cmd[0] == "go":
                kwargs = {}
                if len(cmd) > 1:
                    args = cmd[1].split()
                    if args[0] == "infinite":
                        kwargs["inifinite"] = True
                    else:
                        if args[0] == "ponder":
                            kwargs["ponder"] = True
                            args = args[1:]
                        for i in range(0, len(args) - 1, 2):
                            if args[i] in ["btime", "wtime", "byoyomi", "binc", "winc", "nodes"]:
                                kwargs[args[i]] = int(args[i + 1])

                self.set_limits(**kwargs)
                # save limits and elapsed time for ponderhit.
                last_limits = kwargs
                need_print_bestmove = ("ponder" not in kwargs) and ("infinite" not in kwargs)

                def go_and_print_bestmove():
                    bestmove, ponder_move = self.go()
                    if need_print_bestmove:
                        print(f"bestmove {bestmove}"
                              (f" ponder {ponder_move}" if ponder_move else ""),
                              flush=True
                              )
                    return bestmove, ponder_move

                self.future = self.executor.submit(go_and_print_bestmove)

            elif cmd[0] == "stop":
                # ponderで検討していた相手の手と実際の相手の手が違った場合
                need_print_bestmove = False
                self.stop()
                bestmove, _ = self.future.result()
                print(f"bestmove {bestmove}", flush=True)
            elif cmd[0] == "ponderhit":
                # ponderで検討していた相手の手と実際の相手の手が同じ場合
                last_limits["ponder"] = False
                self.ponderhit(last_limits)
                bestmove, ponder_move = self.future.result()
                print(f"bestmove {bestmove}" + (f" ponder {ponder_move}" if ponder_move else ""),
                      flush=True
                      )
            elif cmd[0] == "quit":
                self.quit()
                break
