import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer
from logging import getLogger

import hydra
import numpy as np
import torch
from optree import tree_map

from mode.evaluation.utils import get_default_mode_and_env

logger = getLogger(__name__)

agent = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def to_device(agent):
    def fn(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        if isinstance(x, torch.Tensor):
            return x.to(agent.device).to(agent.dtype)
        return x
    return fn


class AgentHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        request = pickle.loads(post_data)  # noqa: S301

        mapper = to_device(agent)

        method = request.get("method")
        args = tree_map(mapper, request.get("args", []))
        kwargs = tree_map(mapper, request.get("kwargs", {}))

        try:
            if method == "__shutdown__":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(pickle.dumps({"result": "shutdown"}))
                raise KeyboardInterrupt  # To stop the server

            if method == "__call__":
                result = agent(*args, **kwargs)
            else:
                result = getattr(agent, method)(*args, **kwargs)

            self.send_response(200)
            self.end_headers()
            self.wfile.write(pickle.dumps({"result": result}))
        except Exception:
            logger.exception("Error handling request")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(pickle.dumps({"error": "there was a problem"}))


def start_server(agent_instance, host="localhost", port=6000):
    global agent  # noqa: PLW0603
    agent = agent_instance
    server = HTTPServer((host, port), AgentHandler)
    logger.info(f"starting server at http://{host}:{port}")  # noqa: G004
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down server")
        server.server_close()


@hydra.main(config_path=str("../../conf"), config_name="agent_proxy")
def main(cfg):
    logger.info("loading agent")

    agent = hydra.utils.instantiate(cfg.model)
    
    agent = agent.to(cfg.device)
    agent.eval()

    # model, env, _, lang_embeddings = get_default_mode_and_env(
    #     cfg.train_folder,
    #     cfg.dataset_path,
    #     cfg.checkpoint,
    #     env=42,
    #     lang_embeddings=None,
    #     eval_cfg_overwrite=cfg.eval_cfg_overwrite,
    #     device_id=cfg.device,
    # )
    # model = model.to(cfg.device)
    # model.eval()
    # agent=model

    start_server(agent, cfg.host, cfg.port)


if __name__ == "__main__":
    main()
