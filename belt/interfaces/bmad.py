from belt.input import RFCavity, Bend, DriftTube, Exit, WriteBeam


def belt_element_from_tao(tao, ele_id, beam_radius=0.001):
    head = tao.ele_head(ele_id)
    attrs = tao.ele_gen_attribs(ele_id)
    name = head["name"]
    L = attrs.get("L", 0)

    key = head["key"].lower()

    if key == "lcavity":
        ele = RFCavity(
            length=attrs["L"],
            beam_radius=beam_radius,
            gradient=attrs["GRADIENT"],
            frequency=attrs["RF_FREQUENCY"],
            phase_deg=attrs["PHI0"] * 360,
            name=name,
        )
    elif key in "marker" and (name.startswith("BEG") or name.startswith("END")):
        ele = WriteBeam(iwrite=666, name=name)

    elif key in ("sbend", "rbend"):
        ele = Bend(
            length=L,
            beam_radius=beam_radius,
            angle=attrs["ANGLE"],
            name=name,
        )

    elif L == 0:
        ele = None

    else:
        ele = DriftTube(length=L, beam_radius=beam_radius, name=name)

    return ele


def belt_lattice_from_tao(tao):
    ixs = tao.lat_list("*", "ele.ix_ele")
    eles = []

    for ele_id in ixs:
        ele = belt_element_from_tao(tao, ele_id)
        if ele is None:
            continue
        eles.append(ele)

    # Handle writes
    iwrite = 202
    for ele in eles:
        if isinstance(ele, WriteBeam):
            print(ele.name, iwrite)
            ele.iwrite = iwrite
            iwrite = iwrite + 2

    # last element
    eles.append(Exit())

    return eles
