from pythonosc import udp_client

class OSC:
  def __init__(self, ip="127.0.0.1", port=5005):
    self.ip = ip
    self.port = port
    self.client = udp_client.SimpleUDPClient(self.ip, self.port)

  def send(self, address, value):
    self.client.send_message(address, value)
    

