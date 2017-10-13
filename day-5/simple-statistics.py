# http://pythontutor.com/live.html

import random

class Sensor():
    def __init__(self,id,name="Basic Sensor"):
        self.name = name
        self.id = id
    def __repr__(self):
        return "Sensor({})".format(self.id)
    def name_sensor(self, name):
        self.name = name

    def measure(self):
        return random.random()

sensor_types = ["Water Sensor", "Air seonsor", "Soil sensor", "Humidity sensor", "Heard sensor"]
# sensors = []
# for i in range(0,len(sensor_types)):
#     sensors.append(Sensor(id=i,name=sensor_types[i]))
sensors = [Sensor(id=i,name=sensor_name) for i,sensor_name in enumerate(sensor_types)]

print(sensors)


class SeriesAnalysis():
    def __init__(self,sensor):
        self.sensor = sensor
        self.measurements = []

    def create_series(self,num_measurements=1):
        for i in range(num_measurements):
            self.measurements.append(self.sensor.measure())
    def get_basic_statistics(self):
        return basic_statistics(self.measurements)

class basic_statistics:
    def __init__(self, measurements):
        self.measurements = measurements
    def maximum(self):
        return max(self.measurements)

    def average(self):
        return sum(self.measurements) / len(self.measurements)


del Sensor, sensor_types
series = SeriesAnalysis(sensors[0])
series.create_series(20)
stats = series.get_basic_statistics()
print(basic_statistics.mro())
print(series.create_series)
print(series.__str__)


del SeriesAnalysis, sensors
print("Maximum: ", stats.maximum(), "Average: ", stats.average() )

print(stats.maximum())
#output = [sensor.measure() for sensor in sensors]
#print(output)
