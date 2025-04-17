import numpy as np

from interface import ibpy
from objects.bobject import BObject
from objects.tex_bobject import TexBObject
from utils.constants import DEFAULT_ANIMATION_TIME, FRAME_RATE


class Table(BObject):
    def __init__(self, bob_array: object, **kwargs: object) -> object:
        """
        add objects left-aligned to allow proper alignment inside table
        :param bob_array:
        :param kwargs:

        example
        cube_string=["Anton", "Berta", "Caesar", "Diana","Eda", "Frida", "Gabriela", "Helena"]
         b_cubes = np.array([SimpleTexBObject(cube_string,color='drawing') for cube_string in cube_strings])
        table = Table(b_cubes.reshape((8, 4)), bufferx=0.4, buffery=0.4)
        table.rotate(rotation_euler=[-np.pi / 2, 0, 0], begin_time=0, transition_time=0)
        display.add_text_in(table, line=5, indent=4)
        [table.write_row(i, begin_time=t0 + i * 1, transition_time=1) for i in range(8)]
        t0 += 0.5 + 8 * 1
        """
        self.kwargs=kwargs
        self.array = bob_array

        bufferx = self.get_from_kwargs("bufferx",0.4)
        buffery = self.get_from_kwargs("buffery",0.8)
        head_sep = self.get_from_kwargs('head_sep',0)
        foot_sep = self.get_from_kwargs('foot_sep',0)
        columns = -np.inf
        rows = len(bob_array)
        # add bobs to the table and figure out the numbers of columns
        children=[]
        for row in bob_array:
            if len(row)>columns:
                columns =len(row)
            for col in row:
                if col is not None:
                    children.append(col)
        alignment = self.get_from_kwargs('alignment', ['c'] * columns)

        self.name = self.get_from_kwargs("name","Table")
        super().__init__(children=children,name=self.name,**kwargs)
        # get dimensions of objects
        row_sizes=[-np.inf for row in bob_array]
        col_sizes=[-np.inf for col in range(columns)]

        sizes = []
        for row in range(rows):
            r_sizes = []
            for col in range(columns):
                bob = bob_array[row][col]
                if bob is not None:
                    x_min, y_min, z_min, x_max, y_max, z_max=bob.get_text_bounding_box()
                    dx = x_max-x_min
                    dy = y_max-y_min
                    r_sizes.append((dx,dy))
                    if row_sizes[row]<dy:
                        row_sizes[row]=dy
                    if col_sizes[col]<dx:
                        col_sizes[col]=dx
                else:
                    r_sizes.append((0,0))
            sizes.append(r_sizes)

        #update sizes to the first object's size in case of morphing texts
        for row in range(rows):
            for col in range(columns):
                bob = bob_array[row][col]
                if isinstance(bob,TexBObject):
                    x_min, y_min, z_min, x_max, y_max, z_max = bob.get_first_text_bounding_box()
                    dx = x_max - x_min
                    dy = y_max - y_min
                    sizes[row][col]=((dx,dy))

        #place table data
        width = sum(col_sizes)
        height= sum(row_sizes)

        if len(col_sizes)>1:
            bufferx *= width/(len(col_sizes)-1)
        else:
            bufferx = 0
        if len(row_sizes)>1:
            buffery *= height/(len(row_sizes)-1)
            head_sep *=height/(len(row_sizes)-1)
            foot_sep *=height/(len(row_sizes)-1)
        else:
            buffery=0
            head_sep=0
            foot_sep=0

        current_height=height/2
        for row in range(rows):
            current_width = -width/2
            current_height -= row_sizes[row] / 2
            for col in range(columns):
                align = alignment[col]
                bob = bob_array[row][col]
                if bob is not None:
                    #assuming that all text objects are aligned='left', which is default
                    bob_size = sizes[row][col]
                    # take care of cell alignment
                    if align=='c':
                        current_width+=col_sizes[col]/2
                        ibpy.set_location(bob,location=[current_width-bob_size[0]/2,0,current_height])
                        current_width+=col_sizes[col]/2+bufferx
                    elif align=='l':
                        ibpy.set_location(bob, location=[current_width , 0, current_height])
                        current_width += col_sizes[col] + bufferx
                    else:
                        current_width+=col_sizes[col]+bufferx
                        ibpy.set_location(bob,location=[current_width-bob_size[0],0,current_height])
            current_height -= row_sizes[row] / 2+buffery
            if row==0:
                current_height-=head_sep
            if row==rows-2:
                current_height-=foot_sep
        # make the container appear
        self.appear(begin_time=0,transition_time=0)

    def write_all(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for row in self.array:
            for bob in row:
                if bob is not None:
                    bob.write(begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def write_row(self,row,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        real_bobs=[bob for bob in self.array[row] if bob is not None]
        dt = transition_time/len(real_bobs)
        t0 = begin_time
        for bob in real_bobs:
            t0=bob.write(begin_time=t0,transition_time=dt)
        return begin_time+transition_time

    def disappear_row(self,row, alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        real_bobs = [bob for bob in self.array[row] if bob is not None]
        s = sum([len(bob.letters) for bob in real_bobs])
        dt = np.maximum(1/FRAME_RATE,transition_time/s)
        t0 = begin_time
        for bob in real_bobs:
            for letter in bob.letters:
                t0=letter.disappear(alpha,begin_time=t0,transition_time=dt)
        return begin_time+transition_time


    def write_entry(self,row,col,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        if self.array[row][col] is not None:
            self.array[row][col].write(begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def get_entry(self,row,col):
        return self.array[row][col]

class Table2(BObject):
    def __init__(self, bob_array: object, **kwargs: object) -> object:
        """
        add objects left-aligned to allow proper alignment inside table
        :param bob_array:
        :param kwargs:

        example
        cube_string=["Anton", "Berta", "Caesar", "Diana","Eda", "Frida", "Gabriela", "Helena"]
         b_cubes = np.array([SimpleTexBObject(cube_string,color='drawing') for cube_string in cube_strings])
        table = Table(b_cubes.reshape((8, 4)), bufferx=0.4, buffery=0.4)
        table.rotate(rotation_euler=[-np.pi / 2, 0, 0], begin_time=0, transition_time=0)
        display.add_text_in(table, line=5, indent=4)
        [table.write_row(i, begin_time=t0 + i * 1, transition_time=1) for i in range(8)]
        t0 += 0.5 + 8 * 1
        """
        self.kwargs=kwargs
        self.array = bob_array

        bufferx = self.get_from_kwargs("bufferx",0.4)
        buffery = self.get_from_kwargs("buffery",0.8)
        head_sep = self.get_from_kwargs('head_sep',0)
        foot_sep = self.get_from_kwargs('foot_sep',0)
        columns = -np.inf
        rows = len(bob_array)
        # add bobs to the table and figure out the numbers of columns
        children=[]
        for row in bob_array:
            if len(row)>columns:
                columns =len(row)
            for col in row:
                if col is not None:
                    children.append(col)
        alignment = self.get_from_kwargs('alignment', ['c'] * columns)

        self.name = self.get_from_kwargs("name","Table")
        super().__init__(children=children,name=self.name,**kwargs)
        # get dimensions of objects
        row_sizes=[-np.inf for row in bob_array]
        col_sizes=[-np.inf for col in range(columns)]

        sizes = []
        for row in range(rows):
            r_sizes = []
            for col in range(columns):
                bob = bob_array[row][col]
                if bob is not None:
                    x_min, y_min, z_min, x_max, y_max, z_max=bob.get_text_bounding_box()
                    dx = x_max-x_min
                    dy = y_max-y_min
                    r_sizes.append((dx,dy))
                    if row_sizes[row]<dy:
                        row_sizes[row]=dy
                    if col_sizes[col]<dx:
                        col_sizes[col]=dx
                else:
                    r_sizes.append((0,0))
            sizes.append(r_sizes)

        #update sizes to the first object's size in case of morphing texts
        for row in range(rows):
            for col in range(columns):
                bob = bob_array[row][col]
                if isinstance(bob,TexBObject):
                    x_min, y_min, z_min, x_max, y_max, z_max = bob.get_first_text_bounding_box()
                    dx = x_max - x_min
                    dy = y_max - y_min
                    sizes[row][col]=((dx,dy))

        #place table data
        width = sum(col_sizes)
        height= sum(row_sizes)

        if len(col_sizes)>1:
            bufferx *= width/(len(col_sizes)-1)
        else:
            bufferx = 0
        if len(row_sizes)>1:
            buffery *= height/(len(row_sizes)-1)
            head_sep *=height/(len(row_sizes)-1)
            foot_sep *=height/(len(row_sizes)-1)
        else:
            buffery=0
            head_sep=0
            foot_sep=0

        current_height=height/2
        for row in range(rows):
            current_width = -width/2
            current_height -= row_sizes[row] / 2
            for col in range(columns):
                align = alignment[col]
                bob = bob_array[row][col]
                if bob is not None:
                    #assuming that all text objects are aligned='left', which is default
                    bob_size = sizes[row][col]
                    # take care of cell alignment
                    if align=='c':
                        current_width+=col_sizes[col]/2
                        ibpy.set_location(bob,location=[current_width-bob_size[0]/2,0,current_height])
                        current_width+=col_sizes[col]/2+bufferx
                    elif align=='l':
                        ibpy.set_location(bob, location=[current_width , 0, current_height])
                        current_width += col_sizes[col] + bufferx
                    else:
                        current_width+=col_sizes[col]+bufferx
                        ibpy.set_location(bob,location=[current_width-bob_size[0],0,current_height])
            current_height -= row_sizes[row] / 2+buffery
            if row==0:
                current_height-=head_sep
            if row==rows-2:
                current_height-=foot_sep
        # make the container appear
        self.appear(begin_time=0,transition_time=0)

    def write_all(self,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        for row in self.array:
            for bob in row:
                if bob is not None:
                    bob.write(begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def write_row(self,row,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        real_bobs=[bob for bob in self.array[row] if bob is not None]
        dt = transition_time/len(real_bobs)
        t0 = begin_time
        for bob in real_bobs:
            t0=bob.write(begin_time=t0,transition_time=dt)
        return begin_time+transition_time

    def disappear_row(self,row, alpha=0, begin_time=0, transition_time=DEFAULT_ANIMATION_TIME):
        real_bobs = [bob for bob in self.array[row] if bob is not None]
        s = sum([len(bob.letters) for bob in real_bobs])
        dt = np.maximum(1/FRAME_RATE,transition_time/s)
        t0 = begin_time
        for bob in real_bobs:
            for letter in bob.letters:
                t0=letter.disappear(alpha,begin_time=t0,transition_time=dt)
        return begin_time+transition_time


    def write_entry(self,row,col,begin_time=0,transition_time=DEFAULT_ANIMATION_TIME):
        if self.array[row][col] is not None:
            self.array[row][col].write(begin_time=begin_time,transition_time=transition_time)
        return begin_time+transition_time

    def get_entry(self,row,col):
        return self.array[row][col]