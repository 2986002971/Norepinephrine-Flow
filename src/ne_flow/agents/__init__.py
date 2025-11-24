from ne_flow.agents.crl import CRLAgent
from ne_flow.agents.gcbc import GCBCAgent
from ne_flow.agents.gciql import GCIQLAgent
from ne_flow.agents.gcivl import GCIVLAgent
from ne_flow.agents.hiql import HIQLAgent
from ne_flow.agents.hiql2 import HIQL2Agent
from ne_flow.agents.ne import NE_Agent
from ne_flow.agents.ne_without_high import NE_without_high
from ne_flow.agents.qrl import QRLAgent
from ne_flow.agents.sac import SACAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    hiql=HIQLAgent,
    hiql2=HIQL2Agent,
    neflow=NE_Agent,
    ne_without_high=NE_without_high,
)
