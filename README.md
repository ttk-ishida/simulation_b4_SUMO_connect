# x_intersection_2lane (2-lane X intersection)

This scenario is meant for a single intersection with 2 lanes on each approach (no traffic lights).

## Files
- nodes.nod.xml / edges.edg.xml / types.typ.xml : source network definitions
- make_net.sh : builds net.net.xml using `netconvert`
- routes.rou.xml : 4 straight flows (W<->E, S<->N)
- sumo.sumocfg : SUMO configuration (expects net.net.xml)
- add.add.xml : placeholder for optional add-ons

## Build the network
You must generate `net.net.xml` once:

```bash
cd scenarios/x_intersection_2lane
./make_net.sh
```

## Notes
- Traffic demand is controlled in routes.rou.xml (vehsPerHour).
- Simulation time defaults to 600 s and step-length 0.2 s. Keep these consistent with configs/base.yaml.
