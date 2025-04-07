package main

import (
	"fmt"
	"log"
	"math/cmplx"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/gordonklaus/portaudio"
	"github.com/mjibson/go-dsp/fft"
	"github.com/mjibson/go-dsp/window"
)

const (
	sampleRate      = 44100 
	bufferSize      = 1024  
	maxBarHeight    = 30  
	refreshInterval = 50   
	numBinsToShow   = 64
)

func main() {
	fmt.Println("Initializing PortAudio...")
	if err := portaudio.Initialize(); err != nil {
		log.Fatalf("PortAudio initialization failed: %v\n", err)
	}
	defer portaudio.Terminate() 

	inputDevice, err := portaudio.DefaultInputDevice()
	if err != nil {
		log.Fatalf("Failed to get default input device: %v\n", err)
	}
	fmt.Printf("Using input device: %s\n", inputDevice.Name)

	audioBuffer := make([]float32, bufferSize)

	streamParams := portaudio.StreamParameters{
		Input: portaudio.StreamDeviceParameters{
			Device:   inputDevice,
			Channels: 1,
			Latency:  inputDevice.DefaultLowInputLatency,
		},
		SampleRate:      sampleRate,
		FramesPerBuffer: bufferSize,
	}

	stream, err := portaudio.OpenStream(streamParams, audioBuffer)
	if err != nil {
		log.Fatalf("Failed to open stream: %v\n", err)
	}
	defer stream.Close() 
 
	fmt.Println("Starting audio stream...")
	if err := stream.Start(); err != nil {
		log.Fatalf("Failed to start stream: %v\n", err)
	}
	defer stream.Stop() 

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	done := make(chan bool, 1)

	go func() {
		<-sigChan 
		fmt.Println("\nReceived shutdown signal, stopping...")
		done <- true
	}()

	fftBuffer := make([]float64, bufferSize)
	hannWindow := window.Hann(bufferSize)

	fmt.Println("Starting Spectrum Analyzer (Press Ctrl+C to stop)...")
	ticker := time.NewTicker(refreshInterval * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-done:
			fmt.Println("Exiting.")
			return
		case <-ticker.C:
			err := stream.Read()
			if err != nil {
				
                log.Printf("Warning/Error reading stream: %v\n", err)

			}

			for i := 0; i < bufferSize; i++ {
				fftBuffer[i] = float64(audioBuffer[i]) * hannWindow[i]
			}

			fftResult := fft.FFTReal(fftBuffer)

			magnitudes := make([]float64, bufferSize/2+1)
			maxMagnitude := 0.0 
			for i := range magnitudes {
				magnitudes[i] = cmplx.Abs(fftResult[i])
				if magnitudes[i] < 0.01 {
					magnitudes[i] = 0
				}
				if magnitudes[i] > maxMagnitude {
					maxMagnitude = magnitudes[i]
				}
			}

			clearConsole()
			fmt.Printf("Spectrum (Max Magnitude: %.4f):\n", maxMagnitude)

			freqResolution := float64(sampleRate) / float64(bufferSize)

            step := len(magnitudes) / numBinsToShow
            if step == 0 { step = 1 }

			for i := 0; i < len(magnitudes); i += step {
                if i >= len(magnitudes) { break } 
				mag := magnitudes[i]
				normalizedMag := 0.0
				if maxMagnitude > 0 {
					normalizedMag = mag / maxMagnitude
				}

				barHeight := int(normalizedMag * float64(maxBarHeight))
				if barHeight > maxBarHeight {
					barHeight = maxBarHeight
				}
				if barHeight < 0 {
					barHeight = 0
				}

				bar := strings.Repeat("#", barHeight)

				freq := float64(i) * freqResolution
				fmt.Printf("%6.0f Hz: [%-*s] %.4f\n", freq, maxBarHeight, bar, mag)

                if i / step >= numBinsToShow -1 {
                    break
                }
			}
             fmt.Printf("Displayed bins: %d (Step: %d) | Freq Resolution: %.2f Hz/bin\n", numBinsToShow, step, freqResolution)

		} 
	} 
}


func clearConsole() {
	fmt.Print("\033[H\033[2J")
}