import csv
import os
import sys
import math

from utils import create_directory, dump_pickle, merge_events

class eICUData:

    def __init__(
            self,
            icu_id,
            admission_id,
            patient_id,
            icu_duration,
            hospital_id,
            mortality,
            readmission,
            age,
            gender,
            ethnicity,
    ):
        self.icu_id = icu_id  # str
        self.admission_id = admission_id  # str
        self.patient_id = patient_id  # str
        self.icu_duration = icu_duration  # int
        self.hospital_id = hospital_id  # int
        self.mortality = mortality  # bool, end of icu stay mortality
        self.readmission = readmission  # bool, 15-day icu readmission
        self.age = age  # int
        self.gender = gender  # str
        self.ethnicity = ethnicity  # str

        self.region = None  # str

        # list of tuples (timestamp in min (int), type (str), list of codes (str))
        self.pasthistory = []
        self.admissiondx = []
        self.admissiondrug = []
        self.diagnosis = []
        self.treatment = []
        self.medication = []
        self.lab = []
        self.physicalexam = []

        self.trajectory = []  # (list of timestamps in hour (int), list of types (str), list of list of codes (str))

    def __repr__(self):
        return f"ICU ID-{self.icu_id} ({self.icu_duration} min): " \
               f"mortality-{self.mortality}, " \
               f"readmission-{self.readmission}"


def process_patient(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    admission_dict = {}
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        icu_timestamp = -int(line["hospitaladmitoffset"])  # w.r.t. hospital admission
        admission_id = line["patienthealthsystemstayid"]

        if admission_id not in admission_dict:
            admission_dict[admission_id] = []
        admission_dict[admission_id].append((icu_timestamp, icu_id))

    inff.close()
    print("")

    admission_dict_sorted = {}
    for admission_id, time_icu_tuples in admission_dict.items():
        admission_dict_sorted[admission_id] = sorted(time_icu_tuples)

    icu_readmission_dict = {}
    next_icu_id_dict = {}
    for admission_id, time_icu_tuples in admission_dict_sorted.items():
        for idx, time_icu_tuple in enumerate(time_icu_tuples[:-1]):
            curr_icu_timestamp = time_icu_tuple[0]
            curr_icu_id = time_icu_tuple[1]
            next_icu_timestamp = time_icu_tuples[idx + 1][0]
            next_icu_id = time_icu_tuples[idx + 1][1]
            if next_icu_timestamp - curr_icu_timestamp <= 15 * 24 * 60:
                # re-admitted to ICU within 15 days
                icu_readmission_dict[curr_icu_id] = True
                next_icu_id_dict[curr_icu_id] = next_icu_id
            else:
                icu_readmission_dict[curr_icu_id] = False
                next_icu_id_dict[curr_icu_id] = ""
        last_icu_id = time_icu_tuples[-1][1]
        icu_readmission_dict[last_icu_id] = False
        next_icu_id_dict[last_icu_id] = ""

    excluded_icu_duration = 0
    excluded_age = 0

    inff = open(infile, "r")
    count = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        admission_id = line["patienthealthsystemstayid"]
        patient_id = line["uniquepid"]
        icu_duration = int(line["unitdischargeoffset"])
        # exclude: icu duration > 10 days or < 12 hours
        if (icu_duration > 10 * 24 * 60) or (icu_duration < 12 * 60):
            excluded_icu_duration += 1
            continue
        hospital_id = line["hospitalid"]
        age = line["age"]
        try:
            age = int(age)
        # exclude: cohort age < 18, > 89, or unknown
        except ValueError:
            assert age in ["> 89", ""]
            excluded_age += 1
            continue
        if age < 18:
            excluded_age += 1
            continue
        discharge_status = line["unitdischargestatus"]
        mortality = True if discharge_status == "Expired" else False
        readmission = icu_readmission_dict[icu_id]
        gender = line["gender"]
        ethnicity = line["ethnicity"]

        icu_stay = eICUData(
            icu_id=icu_id,
            admission_id=admission_id,
            patient_id=patient_id,
            icu_duration=icu_duration,
            hospital_id=hospital_id,
            mortality=mortality,
            readmission=readmission,
            age=age,
            gender=gender,
            ethnicity=ethnicity
        )

        if icu_stay in icu_stay_dict:
            print("Duplicate ICU ID!")
            sys.exit(0)
        icu_stay_dict[icu_id] = icu_stay

        count += 1
    inff.close()
    print("")
    print("ICU stays excluded due to icu duration: %d" % excluded_icu_duration)
    print("ICU stays excluded due to age: %d" % excluded_age)
    print("ICU stays included: %d" % len(icu_stay_dict))

    return icu_stay_dict


def process_pasthistory(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    missing_icu_id = 0
    out_of_icu_stay = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        timestamp = int(line["pasthistoryoffset"])
        id = line["pasthistorypath"].lower()

        if icu_id not in icu_stay_dict:
            missing_icu_id += 1
            continue

        if timestamp > icu_stay_dict[icu_id].icu_duration:
            out_of_icu_stay += 1
            continue

        icu_stay_dict[icu_id].pasthistory.append((timestamp, "pasthistory", id))
        count += 1
    inff.close()

    print("")
    print("pasthistory without ICU ID: %d" % missing_icu_id)
    print("pasthistory out of ICU stay: %d" % out_of_icu_stay)

    return icu_stay_dict


def post_process_pasthistory(icu_stay_dict):
    for icu_id, icu_stay in icu_stay_dict.items():
        icu_stay.pasthistory = merge_events(icu_stay.pasthistory)
    return icu_stay_dict


def process_admissiondx(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    missing_icu_id = 0
    out_of_icu_stay = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        timestamp = int(line["admitdxenteredoffset"])
        id = line["admitdxpath"].lower()

        if icu_id not in icu_stay_dict:
            missing_icu_id += 1
            continue

        if timestamp > icu_stay_dict[icu_id].icu_duration:
            out_of_icu_stay += 1
            continue

        icu_stay_dict[icu_id].admissiondx.append((timestamp, "admissiondx", id))
        count += 1
    inff.close()

    print("")
    print("admissiondx without ICU ID: %d" % missing_icu_id)
    print("admissiondx out of ICU stay: %d" % out_of_icu_stay)

    return icu_stay_dict


def post_process_admissiondx(icu_stay_dict):
    for icu_id, icu_stay in icu_stay_dict.items():
        icu_stay.admissiondx = merge_events(icu_stay.admissiondx)
    return icu_stay_dict


def process_admissiondrug(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    missing_icu_id = 0
    out_of_icu_stay = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        timestamp = int(line["drugoffset"])
        id = line["drugname"].lower()

        if icu_id not in icu_stay_dict:
            missing_icu_id += 1
            continue

        if timestamp > icu_stay_dict[icu_id].icu_duration:
            out_of_icu_stay += 1
            continue

        icu_stay_dict[icu_id].admissiondrug.append((timestamp, "admissiondrug", id))
        count += 1
    inff.close()

    print("")
    print("admissiondrug without ICU ID: %d" % missing_icu_id)
    print("admissiondrug out of ICU stay: %d" % out_of_icu_stay)

    return icu_stay_dict


def post_process_admissiondrug(icu_stay_dict):
    for icu_id, icu_stay in icu_stay_dict.items():
        icu_stay.admissiondrug = merge_events(icu_stay.admissiondrug)
    return icu_stay_dict


def process_diagnosis(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    missing_icu_id = 0
    out_of_icu_stay = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        timestamp = int(line["diagnosisoffset"])
        id = line["diagnosisstring"].lower()

        if icu_id not in icu_stay_dict:
            missing_icu_id += 1
            continue

        if timestamp > icu_stay_dict[icu_id].icu_duration:
            out_of_icu_stay += 1
            continue

        icu_stay_dict[icu_id].diagnosis.append((timestamp, "diagnosis", id))
        count += 1
    inff.close()

    print("")
    print("diagnosis without ICU ID: %d" % missing_icu_id)
    print("diagnosis out of ICU stay: %d" % out_of_icu_stay)

    return icu_stay_dict


def post_process_diagnosis(icu_stay_dict):
    for icu_id, icu_stay in icu_stay_dict.items():
        icu_stay.diagnosis = sorted(icu_stay.diagnosis)
    return icu_stay_dict


def process_treatment(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    missing_icu_id = 0
    out_of_icu_stay = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        timestamp = int(line["treatmentoffset"])
        id = line["treatmentstring"].lower()

        if icu_id not in icu_stay_dict:
            missing_icu_id += 1
            continue

        if timestamp > icu_stay_dict[icu_id].icu_duration:
            out_of_icu_stay += 1
            continue

        icu_stay_dict[icu_id].treatment.append((timestamp, "treatment", id))
        count += 1
    inff.close()

    print("")
    print("treatment without ICU ID: %d" % missing_icu_id)
    print("treatment out of ICU stay: %d" % out_of_icu_stay)

    return icu_stay_dict


def post_process_treatment(icu_stay_dict):
    for icu_id, icu_stay in icu_stay_dict.items():
        icu_stay.treatment = sorted(icu_stay.treatment)
    return icu_stay_dict


def process_medication(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    missing_icu_id = 0
    out_of_icu_stay = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        timestamp = int(line["drugstartoffset"])
        id = line["drugname"].lower()

        if icu_id not in icu_stay_dict:
            missing_icu_id += 1
            continue

        if timestamp > icu_stay_dict[icu_id].icu_duration:
            out_of_icu_stay += 1
            continue

        icu_stay_dict[icu_id].medication.append((timestamp, "medication", id))
        count += 1
    inff.close()

    print("")
    print("medication without ICU ID: %d" % missing_icu_id)
    print("medication out of ICU stay: %d" % out_of_icu_stay)

    return icu_stay_dict


def post_process_medication(icu_stay_dict):
    for icu_id, icu_stay in icu_stay_dict.items():
        icu_stay.medication = sorted(icu_stay.medication)
    return icu_stay_dict


def process_lab(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    missing_icu_id = 0
    out_of_icu_stay = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        timestamp = int(line["labresultoffset"])
        id = line["labname"].lower()

        if icu_id not in icu_stay_dict:
            missing_icu_id += 1
            continue

        if timestamp > icu_stay_dict[icu_id].icu_duration:
            out_of_icu_stay += 1
            continue

        icu_stay_dict[icu_id].lab.append((timestamp, "lab", id))
        count += 1
    inff.close()

    print("")
    print("lab without ICU ID: %d" % missing_icu_id)
    print("lab out of ICU stay: %d" % out_of_icu_stay)

    return icu_stay_dict


def post_process_lab(icu_stay_dict):
    for icu_id, icu_stay in icu_stay_dict.items():
        icu_stay.lab = sorted(icu_stay.lab)
    return icu_stay_dict


def process_physicalexam(infile, icu_stay_dict):
    inff = open(infile, "r")
    count = 0
    missing_icu_id = 0
    out_of_icu_stay = 0
    for line in csv.DictReader(inff):
        if count % 10000 == 0:
            sys.stdout.write("%d\r" % count)
            sys.stdout.flush()

        icu_id = line["patientunitstayid"]
        timestamp = int(line["physicalexamoffset"])
        id = line["physicalexampath"].lower()

        if icu_id not in icu_stay_dict:
            missing_icu_id += 1
            continue

        if timestamp > icu_stay_dict[icu_id].icu_duration:
            out_of_icu_stay += 1
            continue

        icu_stay_dict[icu_id].physicalexam.append((timestamp, "physicalexam", id))
        count += 1
    inff.close()

    print("")
    print("physicalexam without ICU ID: %d" % missing_icu_id)
    print("physicalexam out of ICU stay: %d" % out_of_icu_stay)

    return icu_stay_dict


def post_process_physicalexam(icu_stay_dict):
    for icu_id, icu_stay in icu_stay_dict.items():
        icu_stay.physicalexam = merge_events(icu_stay.physicalexam)
    return icu_stay_dict


def process_hospital(infile, icu_stay_dict):
    inff = open(infile, "r")
    hospital_dict = {}
    for line in csv.DictReader(inff):
        hospital_id = line["hospitalid"]
        region = line["region"]
        if region == "":
            region = "Unknown"
        hospital_dict[hospital_id] = region
    inff.close()

    for icu_id, icu_stay in icu_stay_dict.items():
        icu_stay.region = hospital_dict[icu_stay.hospital_id]

    return icu_stay_dict


def post_process(icu_stay_dict, min_len=3, max_len=256):
    min_cut, max_cut = 0, 0
    ret_icu_stay_dict = {}
    for icu_id, icu_stay in icu_stay_dict.items():
        # merge codes
        merged = sorted(
            icu_stay.pasthistory +
            icu_stay.admissiondx +
            icu_stay.admissiondrug +
            icu_stay.diagnosis +
            icu_stay.treatment +
            icu_stay.medication +
            icu_stay.lab +
            icu_stay.physicalexam
        )
        timestamps = [math.ceil((item[0] + 1e-6) / 60) for item in merged]  # min -> hour
        timestamps = [t if t > 0 else 1 for t in timestamps]  # resolve negative timestamp
        types = [item[1] for item in merged]
        codes = [item[2] for item in merged]

        # make prediction at 12 hours
        num_valid_codes = sum([1 for t in timestamps if t <= 12])
        if num_valid_codes < min_len:
            min_cut += 1
            continue
        if num_valid_codes > max_len:
            max_cut += 1
            continue

        timestamps = timestamps[:num_valid_codes]
        types = types[:num_valid_codes]
        codes = codes[:num_valid_codes]

        icu_stay.trajectory = (timestamps, types, codes)

        ret_icu_stay_dict[icu_id] = icu_stay

    print("ICU stays excluded due to min cut: %d" % min_cut)
    print("ICU stays excluded due to max cut: %d" % max_cut)
    print("ICU stays remaining: %d" % len(ret_icu_stay_dict))

    return ret_icu_stay_dict


def main():
    data_path = "/cis/home/xhan56/code/clinical-highmmt/src/datasets/eicu"
    input_path = os.path.join(data_path, "physionet.org/files/eicu-crd/2.0")
    # output_path = os.path.join(data_path, "processed")
    output_path = "/cis/home/xhan56/code/clinical-highmmt"
    create_directory(output_path)

    patient_file = input_path + "/patient.csv"
    pasthistory_file = input_path + "/pastHistory.csv"
    admissiondx_file = input_path + "/admissionDx.csv"
    admissiondrug_file = input_path + "/admissionDrug.csv"
    diagnosis_file = input_path + "/diagnosis.csv"
    treatment_file = input_path + "/treatment.csv"
    medication_file = input_path + "/medication.csv"
    lab_file = input_path + "/lab.csv"
    physicalexam_file = input_path + "/physicalExam.csv"
    hospital_file = input_path + "/hospital.csv"

    icu_stay_dict = {}
    print("Processing patient.csv")
    icu_stay_dict = process_patient(patient_file, icu_stay_dict)

    # print("Processing pastHistory.csv")
    # icu_stay_dict = process_pasthistory(pasthistory_file, icu_stay_dict)
    # print("Post-processing pastHistory")
    # icu_stay_dict = post_process_pasthistory(icu_stay_dict)
    # print("Processing admissionDx.csv")
    # icu_stay_dict = process_admissiondx(admissiondx_file, icu_stay_dict)
    # print("Post-processing admissionDx")
    # icu_stay_dict = post_process_admissiondx(icu_stay_dict)
    # print("Processing admissionDrug.csv")
    # icu_stay_dict = process_admissiondrug(admissiondrug_file, icu_stay_dict)
    # print("Post-processing admissionDrug")
    # icu_stay_dict = post_process_admissiondrug(icu_stay_dict)
    print("Processing diagnosis.csv")
    icu_stay_dict = process_diagnosis(diagnosis_file, icu_stay_dict)
    print("Post-processing diagnosis")
    icu_stay_dict = post_process_diagnosis(icu_stay_dict)
    print("Processing treatment.csv")
    icu_stay_dict = process_treatment(treatment_file, icu_stay_dict)
    print("Post-processing treatment")
    icu_stay_dict = post_process_treatment(icu_stay_dict)
    print("Processing medication.csv")
    icu_stay_dict = process_medication(medication_file, icu_stay_dict)
    print("Post-processing medication")
    icu_stay_dict = post_process_medication(icu_stay_dict)
    print("Processing lab.csv")
    icu_stay_dict = process_lab(lab_file, icu_stay_dict)
    print("Post-processing lab")
    icu_stay_dict = post_process_lab(icu_stay_dict)
    # print("Processing physicalExam.csv")
    # icu_stay_dict = process_physicalexam(physicalexam_file, icu_stay_dict)
    # print("Post-processing physicalExam")
    # icu_stay_dict = post_process_physicalexam(icu_stay_dict)

    print("Processing hospital.csv")
    icu_stay_dict = process_hospital(hospital_file, icu_stay_dict)
    print("Post-processing")
    icu_stay_dict = post_process(icu_stay_dict)

    dump_pickle(icu_stay_dict, os.path.join(output_path, "icu_stay_dict.pkl"))


if __name__ == "__main__":
    main()
