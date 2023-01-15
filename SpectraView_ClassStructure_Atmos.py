import os
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from random import randint
from matplotlib.widgets import Cursor
import scipy.signal as signal
from scipy.optimize import curve_fit
from mie import *

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def add_extra_peak_wl(exp_peak_wl, wavelengths_exp, index_peaks, ExtraPeak):
    ExtraPeakIndex=[]
    ExtraPeakIndex = find_nearest(wavelengths_exp,ExtraPeak)
    exp_peak_wl = np.append(ExtraPeak,exp_peak_wl)
    index_peaks = np.append(ExtraPeakIndex,index_peaks)
    exp_peak_wl=np.sort(exp_peak_wl)
    index_peaks=np.sort(index_peaks)
    return exp_peak_wl, index_peaks

def real_peak(intensities, peak, wavelengths, width=20, test=False):
    """ intensities: list of intensities we are searching for peaks in
    peak: Position of peak found using detect_peaks
    returns: maximum of gaussian fitted around detected peak"""

    try:
        y = intensities[peak - width : peak + width]
        x = wavelengths[peak - width : peak + width]
        interpolated_x = np.linspace(np.min(x), np.max(x), 1000)

        n = len(x)
        mean = sum(x) / n
        sigma = sum(y * (x - mean) ** 2) / n

        def gaus(x, a, x0, sigma):
            return a * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))

        popt, pcov = curve_fit(gaus, x, y, p0=[np.max(y), 1, 2], maxfev=5000)

        if test:
            import matplotlib.pyplot as plt

            plt.plot(x, y, "b+:", color="red")
            plt.plot(
                interpolated_x, gaus(interpolated_x, *popt), color="black", linewidth=2
            )
            # plt.legend()
            plt.xlabel("Wavelength")
            plt.ylabel("Intensity")
            plt.show()

        real_peak = interpolated_x[np.argmax(gaus(interpolated_x, *popt))]
        # print(real_peak)
    except RuntimeError as e:
        print("Runtime Error on {}".format(wavelengths[peak]))
        print(e)
        real_peak = wavelengths[peak]

    except TypeError as e:
        print("TypeError on {}".format(wavelengths[peak]))
        print(e)
        real_peak = wavelengths[peak]

    except ValueError as e:
        # print("ValueError on {}".format(wavelengths[peak]))
        # print(e)
        real_peak = wavelengths[peak]
    return real_peak


def peak_finding(inten_exp, wavelengths_exp, prominence, distance, height, width, WFwave, WF, WidthCutOff=2000000000000):
    index_peaks, width_data = signal.find_peaks(inten_exp, width=width, prominence=prominence, distance=distance, height=height)
    widths = np.ceil(width_data["widths"])
    exp_peak_wl = []
    for peak, width in zip(index_peaks, widths):
        if wavelengths_exp[peak] >= WFwave[1]:
            try:
                exp_peak_wl.append(real_peak(inten_exp, peak, wavelengths_exp, width=np.int_(width*WF[2])))
            except IndexError as e:
                print("WARNING: In file {}, peak {} could not be fitted with a gaussian. Taking approximation".format(find_number(file_), i))
                exp_peak_wl.append(wavelengths_exp[peak])
        elif wavelengths_exp[peak] < WFwave[1] and wavelengths_exp[peak] > WFwave[0]:
            try:
                exp_peak_wl.append(real_peak(inten_exp, peak, wavelengths_exp, width=np.int_(width*WF[1])))
            except IndexError as e:
                print("WARNING: In file {}, peak {} could not be fitted with a gaussian. Taking approximation".format(find_number(file_), i))
                exp_peak_wl.append(wavelengths_exp[peak])
        else:
            try:
                exp_peak_wl.append(real_peak(inten_exp, peak, wavelengths_exp, width=np.int_(width*WF[0])))
            except IndexError as e:
                print("WARNING: In file {}, peak {} could not be fitted with a gaussian. Taking approximation".format(find_number(file_), i))
                exp_peak_wl.append(wavelengths_exp[peak])
    return exp_peak_wl, index_peaks

def adaptive_baseline(inten, coarseness):
    """Takes in intensity spectra and required coarseness of adaptive baseline.
    Returns intensity spectrum with adaptive baseline subtracted"""
    baseline = []
    zero = []
    average = []
    num_points = len(inten)
    interval_size = int(coarseness / 200 * num_points)
    extendinten1=np.full(interval_size,inten[0])
    extendinten2=np.full(interval_size,inten[len(inten)-1])
    intenextended=np.concatenate((extendinten1,inten,extendinten2))    
    for i in range(0, len(inten)+interval_size):
        # Step 1: 0th percentile filter
        zero_percentile = np.percentile(intenextended[i:i+interval_size], 0, axis=0,)
        zero.append(zero_percentile)
    for i in range(0, len(inten)):
        # Step 2: Moving average smoothing
        average_smoothed = np.mean(zero[i:i+interval_size])
        baseline.append(average_smoothed)
    baseline_divided_inten = inten/baseline - 1
    return baseline_divided_inten, baseline

class MainApplication:
    def __init__(self, master):
        self.master = master
        self.colorpeaks=0
        " Create Each Frame "    
        self.frame_a = tk.Frame(self.master,width=1000, height=100, relief=tk.RIDGE, borderwidth=5)
        self.frame_b = tk.Frame(self.master,width=200, height=750, relief=tk.RIDGE, borderwidth=5)
        self.frame_c = tk.Frame(self.master,width=1000, height=750, relief=tk.RIDGE, borderwidth=5)
        self.frame_d = tk.Frame(self.master,width=200, height=100, relief=tk.RIDGE, borderwidth=5)
        
        "Fill Frame A"
        self.FileName_Entry = tk.Entry(self.frame_a, width=145)
        self.FileName_Entry.place(x=0, y=20)
        self.FileName_Entry.insert(0, Filename)
        self.BGPath_Entry = tk.Entry(self.frame_a, width=145)
        self.BGPath_Entry.place(x=0, y=40)
        self.BGPath_Entry.insert(0, BGPathFilename)
        self.PeakPath_Entry = tk.Entry(self.frame_a, width=5)
        self.PeakPath_Entry.place(x=0, y=60)
        self.PeakPath_Entry.insert(0, "1")
        self.LoadFile_Button = tk.Button(self.frame_a,text="Load File",command=self.handle_load_click)
        self.LoadFile_Button.place(x=900, y=25)
        self.LoadPeak_Button = tk.Button(self.frame_a,text="Load Peak",command=self.handle_loadpeaks_click)
        self.LoadPeak_Button.place(x=900, y=52)

        "Fill Frame B"

        self.Coarseness_Entry = tk.Entry(self.frame_b, width=15)
        self.Coarseness_Entry.place(x=80, y=10)
        self.Coarseness_Entry.insert(0, "10")
        self.Coarseness_Label = tk.Label(self.frame_b, text="Coarseness")
        self.Coarseness_Label.place(x=10, y=10)
        self.Filenumber_Entry = tk.Entry(self.frame_b, width=15)
        self.Filenumber_Entry.place(x=80, y=40)
        self.Filenumber_Entry.insert(0, "1")
        self.Filenumber_Label = tk.Label(self.frame_b, text="Filenumber")
        self.Filenumber_Label.place(x=10, y=40)
        self.Minwave_Entry = tk.Entry(self.frame_b, width=15)
        self.Minwave_Entry.place(x=80, y=70)
        self.Minwave_Entry.insert(0, "0.3")
        self.Minwave_Label = tk.Label(self.frame_b, text="Min Wave")
        self.Minwave_Label.place(x=10, y=70)
        self.Maxwave_Entry = tk.Entry(self.frame_b, width=15)
        self.Maxwave_Entry.place(x=80, y=100)
        self.Maxwave_Entry.insert(0, "0.7")
        self.Maxwave_Label = tk.Label(self.frame_b, text="Max Wave")
        self.Maxwave_Label.place(x=10, y=100)
        self.SaveArraytoFile_Button = tk.Button(self.frame_b, text="Save Array To File",command=self.handle_savetofile_click)
        self.SaveArraytoFile_Button.place(x=30, y=130)
        self.Peak_Label = tk.Label(self.frame_b, text="Peak = ")
        self.Peak_Label.place(x=30, y=160)
        self.BG_Button = tk.Button(self.frame_b, text="True",command=self.toggleBG)
        self.BG_Button.place(x=80, y=190)
        self.BG_Button.config(text='False')
        self.Smooth_Button = tk.Button(self.frame_b, text="True",command=self.togglesmooth)
        self.Smooth_Button.place(x=80, y=220)
        self.Normalize_Button = tk.Button(self.frame_b, text="True",command=self.togglenormalize)
        self.Normalize_Button.place(x=80, y=250)
        self.Normalize_Button.config(text='False')
        self.Baseline_Button = tk.Button(self.frame_b, text="True",command=self.togglebaseline)
        self.Baseline_Button.place(x=80, y=280)
        self.BG_Label = tk.Label(self.frame_b, text="Subtract BG")
        self.BG_Label.place(x=10, y=190)
        self.Smooth_Label = tk.Label(self.frame_b, text="Smooth")
        self.Smooth_Label.place(x=10, y=220)
        self.Smooth_Entry1 = tk.Entry(self.frame_b, width=15)
        self.Smooth_Entry1.place(x=10, y=460)
        self.Smooth_Entry1.insert(0, "5")
        self.Smooth_Entry2 = tk.Entry(self.frame_b, width=15)
        self.Smooth_Entry2.place(x=10, y=490)
        self.Smooth_Entry2.insert(0, "4")
        self.Normalize_Label = tk.Label(self.frame_b, text="Normalize")
        self.Normalize_Label.place(x=10, y=250)
        self.Baseline_Label = tk.Label(self.frame_b, text="Baseline")
        self.Baseline_Label.place(x=10, y=280)
        self.Theory_Button = tk.Button(self.frame_b, text="Add Theory Spectrum",command=self.handle_theory_click)
        self.Theory_Button.place(x=10, y=310)
        self.A_Entry = tk.Entry(self.frame_b, width=15)
        self.A_Entry.place(x=10, y=340)
        self.A_Entry.insert(0, "1.5000")
        self.B_Entry = tk.Entry(self.frame_b, width=15)
        self.B_Entry.place(x=10, y=370)
        self.B_Entry.insert(0, "3E-3")
        self.C_Entry = tk.Entry(self.frame_b, width=15)
        self.C_Entry.place(x=10, y=400)
        self.C_Entry.insert(0, "4E-4")
        self.r_Entry = tk.Entry(self.frame_b, width=15)
        self.r_Entry.place(x=10, y=430)
        self.r_Entry.insert(0, "0.850")
        self.AddBaselineCurve_Button = tk.Button(self.frame_b, text="Add Baseline Curve",command=self.handle_baselinecurve_click)
        self.AddBaselineCurve_Button.place(x=10, y=520)
        self.WFwave1 = tk.Entry(self.frame_b, width=5)
        self.WFwave1.place(x=10, y=550)
        self.WFwave1.insert(0, "0.355")
        self.WFwave2 = tk.Entry(self.frame_b, width=5)
        self.WFwave2.place(x=50, y=550)
        self.WFwave2.insert(0, "0.365")
        self.WF1 = tk.Entry(self.frame_b, width=5)
        self.WF1.place(x=10, y=580)
        self.WF1.insert(0, "0.1")
        self.WF2 = tk.Entry(self.frame_b, width=5)
        self.WF2.place(x=50, y=580)
        self.WF2.insert(0, "0.2")
        self.WF3 = tk.Entry(self.frame_b, width=5)
        self.WF3.place(x=90, y=580)
        self.WF3.insert(0, "0.3")
        self.PeakFind_Button = tk.Button(self.frame_b, text="Find Peaks",command=self.handle_findpeaks_click)
        self.PeakFind_Button.place(x=10, y=610)
        self.Exp_Prom = tk.Entry(self.frame_b, width=5)
        self.Exp_Prom.place(x=90, y=550)
        self.Exp_Prom.insert(0, "0.05")
        self.Extrawave1 = tk.Entry(self.frame_b, width=5)
        self.Extrawave1.place(x=10, y=640)
        self.Extrawave1.insert(0, "0.321")
        self.Extrawave2 = tk.Entry(self.frame_b, width=5)
        self.Extrawave2.place(x=50, y=640)
        self.Extrawave2.insert(0, "0.323")
        
        "Fill Frame C"

        self.fig, self.ax = plt.subplots(figsize=(9,6.5))
        self.canvas = FigureCanvasTkAgg(self.fig, self.frame_c)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame_c)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        "Fill Frame D"

        self.Quit_Button = tk.Button(self.frame_d, text="Quit", command=self.master.quit)
        self.Quit_Button.place(x=20, y=25)

        "Place Frames in Window"

        self.frame_a.grid(row=0,column=0)
        self.frame_b.grid(row=1,column=1)
        self.frame_c.grid(row=1,column=0)
        self.frame_d.grid(row=0,column=1)

        self.master.bind("<Left>", self.handle_leftarrowpress)
        self.master.bind("<Right>", self.handle_rightarrowpress)
        self.master.bind("<Return>", self.handle_enterpress)
        self.master.bind("<a>", self.handle_apress)
        self.master.bind("<d>", self.handle_dpress)

    def handle_load_click(self):
        " Load in the spectrum and the calibrated wavelength file "
        self.SpectrumPathFilename = self.FileName_Entry.get()
        self.BGPathFilename = self.BGPath_Entry.get()
        self.wavelengths_exp=np.loadtxt(self.BGPathFilename, skiprows =2)[:, 0]/1000
        self.inten_bg=np.loadtxt(self.BGPathFilename, skiprows =2)[:, 1]
        self.min_index=find_nearest(self.wavelengths_exp,float(self.Minwave_Entry.get()))
        self.max_index=find_nearest(self.wavelengths_exp,float(self.Maxwave_Entry.get()))
        self.wavelengths_exp=self.wavelengths_exp[self.min_index:self.max_index+1]
        self.inten_bg=self.inten_bg[self.min_index:self.max_index+1]
        self.filenumber=self.Filenumber_Entry.get()
        self.inten_exp = np.loadtxt(self.SpectrumPathFilename, delimiter=",", skiprows =2, usecols=range(int(self.filenumber)+1))[:, int(self.filenumber)]
        self.inten_exp=self.inten_exp[self.min_index:self.max_index+1]

        " Apply the necessary processing to the file "
        if self.BG_Button.config('text')[-1] == 'True':
            self.inten_exp_raw = self.inten_exp
            self.inten_exp -= self.inten_bg
        if self.Smooth_Button.config('text')[-1] == 'True':
            self.inten_exp = signal.savgol_filter(self.inten_exp,2*int(self.Smooth_Entry1.get())+1,int(self.Smooth_Entry2.get()))
        if self.Baseline_Button.config('text')[-1] == 'True':
            self.inten_exp,_ = adaptive_baseline(self.inten_exp, coarseness=float(self.Coarseness_Entry.get()))
        if self.Normalize_Button.config('text')[-1] == 'True':
            self.inten_exp = self.inten_exp / np.max(self.inten_exp)

        "Fill in the Graph"
        plt.cla()
        self.ax.plot(self.wavelengths_exp, self.inten_exp)
        self.ax.set_xlabel("Wavelength / $\mu m$")
        self.ax.set_ylabel("Intensity / arb. units")
        self.canvas.draw()
        self.toolbar.pack()
        self.canvas.get_tk_widget().pack()
        self.peakwavearray=[]
        self.peakintenarray=[]

        " Add the Cursor "
        self.xposindex=find_nearest(self.wavelengths_exp,np.mean(self.wavelengths_exp))
        self.xpos=self.wavelengths_exp[self.xposindex]
        ypos=self.inten_exp[self.xposindex]
        self.horizontal_line = self.ax.axhline(y=ypos, color='k', lw=0.8, ls='--')
        self.vertical_line = self.ax.axvline(x=self.xpos, color='k', lw=0.8, ls='--')
        self.canvas.draw()

    def handle_loadpeaks_click(self):
        self.PathName=self.FileName_Entry.get()[0:self.FileName_Entry.get().rfind("/")]        
        self.ParticleName=self.FileName_Entry.get()[self.FileName_Entry.get().rfind("/"):]
        self.ParticleName=self.ParticleName.replace('.csv','')
        self.PeakPathFilename = self.PathName + self.ParticleName + "File" + self.Filenumber_Entry.get()+ "Peaks" + self.PeakPath_Entry.get() + ".txt"
        self.filepeakwave=np.loadtxt(self.PeakPathFilename, usecols=range(1))
        self.filepeakinten=np.full(len(self.filepeakwave),0,np.float64)
        for i in range(0, len(self.filepeakwave)):
            self.filepeakinten[i]=self.inten_exp[find_nearest(self.wavelengths_exp,self.filepeakwave[i])]
        self.colorpeaks+=1
        if (self.colorpeaks % 2) == 0:
            self.ax.scatter(self.filepeakwave, self.filepeakinten,c='g',marker='x')
        else:
            self.ax.scatter(self.filepeakwave, self.filepeakinten,c='k',marker='x')
        self.canvas.draw()

    def handle_savetofile_click(self):
        self.peakarray=np.stack((self.peakwavearray,self.peakintenarray))
        self.peakarray=np.transpose(self.peakarray)
        self.peakarray=self.peakarray[np.argsort(self.peakarray[:,0])]
        np.savetxt("peakarray.csv",self.peakarray,delimiter=",")
        """ Save array to a csv or txt file """
        """ Need some way to remove all or some peaks """

    def handle_leftarrowpress(self, event):
        self.xposindex-=1
        self.xpos=self.wavelengths_exp[self.xposindex]
        ypos=self.inten_exp[self.xposindex]
        self.horizontal_line.set_ydata(ypos)
        self.vertical_line.set_xdata(self.xpos)
        self.canvas.draw()
        self.Peak_Label['text'] = " Peak = " + str(np.round(self.xpos,6))

    def handle_rightarrowpress(self, event):
        self.xposindex+=1
        self.xpos=self.wavelengths_exp[self.xposindex]
        ypos=self.inten_exp[self.xposindex]
        self.horizontal_line.set_ydata(ypos)
        self.vertical_line.set_xdata(self.xpos)
        self.canvas.draw()
        self.Peak_Label['text'] = " Peak = " + str(np.round(self.xpos,6))

    def handle_apress(self, event):
        self.xposindex-=50
        self.xpos=self.wavelengths_exp[self.xposindex]
        ypos=self.inten_exp[self.xposindex]
        self.horizontal_line.set_ydata(ypos)
        self.vertical_line.set_xdata(self.xpos)
        self.canvas.draw()
        self.Peak_Label['text'] = " Peak = " + str(np.round(self.xpos,6))

    def handle_dpress(self, event):
        self.xposindex+=50
        self.xpos=self.wavelengths_exp[self.xposindex]
        ypos=self.inten_exp[self.xposindex]
        self.horizontal_line.set_ydata(ypos)
        self.vertical_line.set_xdata(self.xpos)
        self.canvas.draw()
        self.Peak_Label['text'] = " Peak = " + str(np.round(self.xpos,6))

    def handle_enterpress(self, event):
        self.peakwavearray=np.append(self.peakwavearray,self.xpos)
        self.peakintenarray=np.append(self.peakintenarray,self.inten_exp[self.xposindex])
        self.ax.scatter(self.peakwavearray, self.peakintenarray,c='r',marker='x')
        self.canvas.draw()

    def toggleBG(self):
        if self.BG_Button.config('text')[-1] == 'True':
            self.BG_Button.config(text='False')
        else:
            self.BG_Button.config(text='True')

    def togglesmooth(self):
        if self.Smooth_Button.config('text')[-1] == 'True':
            self.Smooth_Button.config(text='False')
        else:
            self.Smooth_Button.config(text='True')

    def togglenormalize(self):
        if self.Normalize_Button.config('text')[-1] == 'True':
            self.Normalize_Button.config(text='False')
        else:
            self.Normalize_Button.config(text='True')

    def togglebaseline(self):
        if self.Baseline_Button.config('text')[-1] == 'True':
            self.Baseline_Button.config(text='False')
        else:
            self.Baseline_Button.config(text='True')
            
    def handle_theory_click(self):
        self.theory_intensities = []
        A=float(self.A_Entry.get())
        B=float(self.B_Entry.get())
        C=float(self.C_Entry.get())
        radius=float(self.r_Entry.get())
        for i in self.wavelengths_exp:
            n_particle = cauchy(i, A, B, C)
            SizeP = calc_size_param(radius, i)
            inten = (i ** 2) * bhmie(SizeP, n_particle, 1.00027, 61, 149, 180)
            self.theory_intensities.append(inten)
        self.theory_intensities = np.array(self.theory_intensities)
        
        " Apply the necessary processing to the file "
        if self.Smooth_Button.config('text')[-1] == 'True':
            self.theory_intensities = signal.savgol_filter(self.theory_intensities,2*int(self.Smooth_Entry1.get())+1,int(self.Smooth_Entry2.get()))
        if self.Baseline_Button.config('text')[-1] == 'True':
            self.theory_intensities,_ = adaptive_baseline(self.theory_intensities, coarseness=float(self.Coarseness_Entry.get()))
        if self.Normalize_Button.config('text')[-1] == 'True':
            self.theory_intensities = self.theory_intensities / np.max(self.theory_intensities)

        "Fill in the Graph"
        self.ax.plot(self.wavelengths_exp, self.theory_intensities)
        self.canvas.draw()

    def handle_baselinecurve_click(self):
        _,baseline = adaptive_baseline(self.inten_exp, coarseness=float(self.Coarseness_Entry.get()))
        self.ax.plot(self.wavelengths_exp, baseline)
        self.canvas.draw()
        
    def handle_findpeaks_click(self):
        "Find Peaks Normally"
        WFwave=[float(self.WFwave1.get()), float(self.WFwave2.get())]
        WF=[float(self.WF1.get()), float(self.WF2.get()), float(self.WF3.get())]
        self.exp_peak_wl, self.index_peaks = peak_finding(
            self.inten_exp,
            self.wavelengths_exp,
            prominence=float(self.Exp_Prom.get()),
            distance=5,
            height=0,
            width=1,
            WFwave=WFwave,
            WF=WF,
        )
        "Find smaller peaks"
        exp_peak_wl_extra, index_peaks_extra = peak_finding(
            self.inten_exp,
            self.wavelengths_exp,
            prominence=0.0075,
            distance=5,
            height=0,
            width=1,
            WFwave=WFwave,
            WF=WF)
        exp_peak_wl_extra = [x for x in exp_peak_wl_extra if (float(self.Extrawave1.get())<=x<=float(self.Extrawave2.get()))]
        for i in range(0,len(exp_peak_wl_extra)):
            self.exp_peak_wl, self.index_peaks = add_extra_peak_wl(self.exp_peak_wl, self.wavelengths_exp, self.index_peaks, exp_peak_wl_extra[i])

        "Get the intensities"
        self.exp_peak_inten=np.full(len(self.exp_peak_wl),0,np.float64)
        for i in range(0, len(self.exp_peak_wl)):
            self.exp_peak_inten[i]=self.inten_exp[self.index_peaks[i]]

        "Add it to the plot"
        self.ax.scatter(self.exp_peak_wl, self.exp_peak_inten,c='r',marker='x')
        self.canvas.draw()

        "Save it to a text file"
        
        self.PathName=self.FileName_Entry.get()[0:self.FileName_Entry.get().rfind("/")]        
        self.ParticleName=self.FileName_Entry.get()[self.FileName_Entry.get().rfind("/"):]
        self.ParticleName=self.ParticleName.replace('.csv','')
        np.savetxt((self.PathName + self.ParticleName + "File" + self.Filenumber_Entry.get()+ "Peaks" + self.PeakPath_Entry.get() + "_FromSV.txt"),self.exp_peak_wl)
        
def main():
    window = tk.Tk()
    window.wm_title("Select Peaks")
    app = MainApplication(window)
    window.mainloop()

if __name__ == "__main__":
    " Parameters "
    BGPathFilename="C:/Users/connor/OneDrive/Documents/PhD/Fitting/Atmospheric Samples/LDLSoff_300g_399_5s_20umsl_Calibrated_ForWC14thFeb.txt"
    Filename="C:/Users/connor/OneDrive/Documents/PhD/Fitting/Atmospheric Samples/AliceHoltUpper/AliceHoltUpper_Spring19/B2/AprilMay19AliceHoltUpper_300g_399_5s_20umsl_B2.csv"
    ParticleName=Filename[0: Filename.find("_")]
    main()
    

