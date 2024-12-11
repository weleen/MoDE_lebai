import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer
from logging import getLogger

import hydra
import numpy as np
import torch
from optree import tree_map

logger = getLogger(__name__)

agent = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def to_device(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


class AgentHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        request = pickle.loads(post_data)  # noqa: S301

        method = request.get("method")
        args = request.get("args", [])
        args = tree_map(to_device, args)
        kwargs = request.get("kwargs", {})
        kwargs = tree_map(to_device, kwargs)

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

    start_server(agent, cfg.host, cfg.port)


if __name__ == "__main__":
    main()
