ดาวน์โหลด https://protege.stanford.edu/

# บทเรียนการสร้างออนโทโลยีของสัตว์โดยใช้ Protege

## ขั้นตอนที่ 1: เข้าโปรแกรม Protege 

1. **เข้าถึง Web Protege**
   - เปิดเว็บเบราว์เซอร์และเข้าไปที่ [Web Protege](https://webprotege.stanford.edu/)
   - หากยังไม่มีบัญชี ให้สมัครบัญชีใหม่

2. **สร้างโปรเจกต์ใหม่**
   - เมื่อเข้าสู่ระบบแล้ว คลิกที่ปุ่ม Create new project
   - เลือก Blank project เพื่อเริ่มโปรเจกต์ใหม่จากศูนย์
   - ตั้งชื่อโปรเจกต์ เช่น AnimalOntology
   - กด Create เพื่อสร้างโปรเจกต์

## ขั้นตอนที่ 2: การสร้างคลาส (Classes) พื้นฐาน

1. **เข้าแท็บ Classes**
   - ในเมนูด้านซ้าย คลิกที่ Classes

2. **สร้างคลาสหลัก "Animal
   - คลิกขวาที่ Thing และเลือก Add subclass
   - ตั้งชื่อคลาสว่า Animal

3. **เพิ่มคลาสย่อยสำหรับประเภทของสัตว์**
   - คลิกขวาที่ Animal และเลือก Add subclass
   - เพิ่มคลาสย่อยต่อไปนี้:
     - Mammal
     - Bird
     - Fish
     - Reptile
     - Amphibian

4. **เพิ่มคลาสเฉพาะสำหรับสัตว์แต่ละชนิด**
   - ภายใต้ Mammal, เพิ่ม:
     - Dog
     - Cat
     - Elephant
   - ภายใต้ Bird, เพิ่ม:
     - Eagle
     - Penguin

## ขั้นตอนที่ 3: การสร้างคุณสมบัติ (Properties)

### 3.1 Object Properties

1. **เข้าแท็บ Object Properties**
   - คลิกที่ Object Properties ในเมนูด้านซ้าย

2. **สร้างคุณสมบัติความสัมพันธ์ระหว่างสัตว์**
   - คลิกที่ปุ่ม + เพื่อเพิ่มคุณสมบัติใหม่
   - ตั้งชื่อคุณสมบัติว่า hasHabitat
     - คำอธิบาย: ระบุถิ่นที่อยู่ของสัตว์
   - เพิ่มคุณสมบัติอื่นๆ เช่น:
     - eats: สัตว์กินอะไร
     - isPredatorOf: เป็นผู้ล่าของสัตว์ใด

### 3.2 Data Properties

1. **เข้าแท็บ Data Properties**
   - คลิกที่ Data Properties

2. **สร้างคุณสมบัติข้อมูลของสัตว์**
   - คลิกที่ปุ่ม +
   - ตั้งชื่อคุณสมบัติว่า hasAge
     - คำอธิบาย: อายุของสัตว์
   - เพิ่มคุณสมบัติอื่นๆ เช่น:
     - hasWeight: น้ำหนักของสัตว์

## ขั้นตอนที่ 4: การกำหนดคุณสมบัติให้กับคลาส

1. **กำหนดขอบเขตของคุณสมบัติ**
   - ใน Object Properties, เลือก hasHabitat
   - ในส่วน Domains and Ranges, ตั้งค่า:
     - **Domain**: Animal
     - **Range**: Habitat (สร้างคลาส Habitat หากยังไม่มี)

2. **ทำเช่นเดียวกันกับคุณสมบัติอื่นๆ**

## ขั้นตอนที่ 5: การเพิ่มอินดิวิเดวล (Individuals)

1. **เข้าแท็บ Individuals**
   - คลิกที่ Individuals

2. **สร้างอินดิวิเดวลสำหรับสัตว์แต่ละตัว**
   - คลิกที่ปุ่ม + เพื่อเพิ่มอินดิวิเดวลใหม่
   - ตั้งชื่ออินดิวิเดวลว่า Tommy และกำหนดคลาส Dog
   - เพิ่มอินดิวิเดวลอื่นๆ เช่น:
     - Kitty: คลาส Cat
     - Dumbo: คลาส Elephant

## ขั้นตอนที่ 6: การกำหนดค่าคุณสมบัติให้กับอินดิวิเดวล

1. **เลือกอินดิวิเดวลที่ต้องการ**
   - เช่น เลือก Tommy

2. **เพิ่มค่าคุณสมบัติ**
   - ในส่วน Relationships เพิ่ม Property
   - เลือก hasAge และระบุค่าเป็น **5** (ปี)
   - เลือก hasWeight และระบุค่าเป็น **20** (กิโลกรัม)
   - เลือก hasHabitat และระบุค่าเป็น Domestic

## ขั้นตอนที่ 7: การทำเหตุผล (Reasoning)

DL Query สำหรับออนโทโลยีสัตว์
1. การค้นหาพื้นฐาน
ค้นหาประเภทสัตว์:
 
# ค้นหาสัตว์ทั้งหมด
Animal

# ค้นหาสัตว์เลี้ยงลูกด้วยนม
Mammal

# ค้นหาสัตว์ปีก
Bird

# ค้นหาสัตว์เลื้อยคลาน
Reptile

# ค้นหาสัตว์ครึ่งบกครึ่งน้ำ
Amphibian
 
2. การค้นหาตามที่อยู่อาศัย (hasHabitat)
 
# ค้นหาสัตว์ที่อาศัยในน้ำ
Animal and (hasHabitat some Water)

# ค้นหาสัตว์ที่อาศัยบนบก
Animal and (hasHabitat some Land)

# ค้นหาสัตว์ที่อาศัยในอากาศ
Animal and (hasHabitat some Air)

# ค้นหาสัตว์ที่อาศัยได้ทั้งน้ำและบก
Animal and (hasHabitat some Water) and (hasHabitat some Land)
 
4. การค้นหาตามคุณสมบัติทางกายภาพ
 
# ค้นหาสัตว์ที่มีอายุมากกว่า 10 ปี
Animal and (hasAge some xsd:integer[> 10])

# ค้นหาสัตว์ที่มีน้ำหนักน้อยกว่า 5 กิโลกรัม
Animal and (hasWeight some xsd:float[< 5.0])

# ค้นหาสัตว์ที่มีขนาดใหญ่
Animal and (hasSize value "large")
 
5. การค้นหาแบบซับซ้อน
 
# ค้นหาสัตว์เลี้ยงลูกด้วยนมที่อาศัยในน้ำ
Mammal and (hasHabitat some Water)

# ค้นหานกที่กินเนื้อ
Bird and (eats some Meat)

# ค้นหาสัตว์เลื้อยคลานที่อาศัยได้ทั้งบกและน้ำ
Reptile and (hasHabitat some Land) and (hasHabitat some Water)

