from ne_flow.agents.crl import CRLAgent
from ne_flow.agents.gcbc import GCBCAgent
from ne_flow.agents.gciql import GCIQLAgent
from ne_flow.agents.gcivl import GCIVLAgent
from ne_flow.agents.hiql import HIQLAgent
from ne_flow.agents.hiql2 import HIQL2Agent
from ne_flow.agents.ne_with_temporal_ensemble import NE_with_temporal_ensemble_Agent
from ne_flow.agents.ne_without_high import NE_without_high
from ne_flow.agents.ne_without_temporal_ensemble import NE_without_temporal_ensemble
from ne_flow.agents.ne_without_topk import NE_without_topk
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
    neflow_withtemporal=NE_with_temporal_ensemble_Agent,
    ne_without_high=NE_without_high,
    neflow_notemporal=NE_without_temporal_ensemble,
    neflow_notopk=NE_without_topk,
)
