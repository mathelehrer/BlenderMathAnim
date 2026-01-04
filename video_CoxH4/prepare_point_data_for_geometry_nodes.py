from mathematics.geometry.field_extensions import QR


if __name__ == '__main__':
    with open("data/v_14400_points.csv","r") as file:
        with open("data/v_14400_ints.csv","w") as out:
            out.write("x1,x2,x3,x4,y1,y2,y3,y4,z1,z2,z3,z4,w1,w2,w3,w4\n")
            for i,line in enumerate(file):
                if i>0:
                    out_line = ""
                    components = line[:-1].split(",")
                    for comp in components:
                        q = QR.parse(comp)
                        out_line+=str(q.to_integer())[1:-1]+","

                    out.write(out_line[:-1]+"\n")