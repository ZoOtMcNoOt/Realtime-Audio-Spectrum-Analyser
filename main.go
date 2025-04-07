package main

import (
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
	"github.com/gordonklaus/portaudio"
	"github.com/gorilla/websocket"
	"github.com/mjibson/go-dsp/fft"
	"github.com/mjibson/go-dsp/window"
)

const (
	sampleRate      = 44100
	bufferSize      = 4096
	refreshInterval = 40
	
	numBars      = 128
	minFreq      = 1000.0    
	maxFreq      = 22050.0   
	dbRange      = 80.0
	
	smoothingFactor = 0.3
	minMagnitudeDb  = -dbRange
	maxMagnitudeDb  = 0.0
	epsilon         = 1e-9
	
	webPort = 8080
)

type logFreqBin struct {
	centerFreq float64
	startBin   int
	endBin     int
}

type SpectrumData struct {
	FrequencyData []float64 `json:"frequencyData"`
	PeakData      []float64 `json:"peakData"`
	MinFreq       float64   `json:"minFreq"`
	MaxFreq       float64   `json:"maxFreq"`
	MinDb         float64   `json:"minDb"`
	MaxDb         float64   `json:"maxDb"`
	Timestamp     int64     `json:"timestamp"`
}

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true 
	},
}

func main() {
	err := portaudio.Initialize()
	if err != nil {
		log.Fatalf("Failed to initialize PortAudio: %v\n", err)
	}
	defer portaudio.Terminate()

	audioBuffer := make([]float32, bufferSize)
	stream, err := portaudio.OpenDefaultStream(1, 0, float64(sampleRate), bufferSize, audioBuffer)
	if err != nil {
		log.Fatalf("Failed to open audio stream: %v\n", err)
	}
	defer stream.Close()

	logFreqBins := calculateLogFreqBins(numBars, minFreq, maxFreq, sampleRate, bufferSize)

	err = stream.Start()
	if err != nil {
		log.Fatalf("Failed to start audio stream: %v\n", err)
	}
	defer stream.Stop()

	broadcast := make(chan SpectrumData)
	
	clients := make(map[*websocket.Conn]bool)
	
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "index.html")
	})
	
	http.HandleFunc("/spectrum", func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("Error upgrading to WebSocket: %v\n", err)
			return
		}
		
		clients[conn] = true
		log.Printf("Client connected: %s\n", conn.RemoteAddr())
		
		defer func() {
			conn.Close()
			delete(clients, conn)
			log.Printf("Client disconnected: %s\n", conn.RemoteAddr())
		}()
		
		for {
			_, _, err := conn.ReadMessage()
			if err != nil {
				break
			}
		}
	})
	
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))
	
	go func() {
		log.Printf("Starting web server on port %d\n", webPort)
		log.Printf("Open http://localhost:%d in your web browser\n", webPort)
		log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", webPort), nil))
	}()
	
	go func() {
		for data := range broadcast {
			for client := range clients {
				err := client.WriteJSON(data)
				if err != nil {
					log.Printf("Error sending to client: %v\n", err)
					client.Close()
					delete(clients, client)
				}
			}
		}
	}()

	fftBuffer := make([]float64, bufferSize)
	hannWindow := window.Hann(bufferSize)
	smoothedDbValues := make([]float64, numBars)
	peakDbValues := make([]float64, numBars)
	peakHoldCounters := make([]int, numBars)
	
	for i := range smoothedDbValues {
		smoothedDbValues[i] = minMagnitudeDb
		peakDbValues[i] = minMagnitudeDb
	}

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	done := make(chan bool, 1)
	
	go func() {
		<-sigChan
		log.Println("Shutting down...")
		close(broadcast)
		done <- true
	}()

	log.Println("Audio analyzer running. Press Ctrl+C to stop.")
	ticker := time.NewTicker(refreshInterval * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			if len(clients) == 0 {
				continue
			}
			
			err := stream.Read()
			if err != nil {
				log.Printf("Error reading audio: %v\n", err)
				continue
			}

			for i := 0; i < bufferSize; i++ {
				fftBuffer[i] = float64(audioBuffer[i]) * hannWindow[i]
			}
			
			fftResult := fft.FFTReal(fftBuffer)
			
			calculateSpectrum(fftResult, smoothedDbValues, peakDbValues, peakHoldCounters, logFreqBins)
			
			specData := SpectrumData{
				FrequencyData: append([]float64{}, smoothedDbValues...),
				PeakData:      append([]float64{}, peakDbValues...),
				MinFreq:       minFreq,
				MaxFreq:       maxFreq,
				MinDb:         minMagnitudeDb,
				MaxDb:         maxMagnitudeDb,
				Timestamp:     time.Now().UnixNano() / int64(time.Millisecond),
			}
			
			broadcast <- specData
		}
	}
}

func calculateSpectrum(fftResult []complex128, smoothedDbValues, peakDbValues []float64, peakHoldCounters []int, logFreqBins []logFreqBin) {
	const peakHoldFrames = 30
	const peakDecayRate = 0.6
	
	for i := 0; i < len(smoothedDbValues); i++ {
		binInfo := logFreqBins[i]
		if binInfo.startBin >= binInfo.endBin || binInfo.startBin >= len(fftResult) {
			continue
		}

		sumMagSq := 0.0
		binCount := 0
		endBin := min(binInfo.endBin, len(fftResult))
		
		for k := binInfo.startBin; k < endBin; k++ {
			mag := cmplx.Abs(fftResult[k])
			sumMagSq += mag * mag  
			binCount++
		}

		avgMag := 0.0
		if binCount > 0 {
			avgMag = math.Sqrt(sumMagSq / float64(binCount))  
		}

		db := 20.0 * math.Log10(math.Max(epsilon, avgMag))
		db = math.Max(minMagnitudeDb, math.Min(maxMagnitudeDb, db))

		smoothedDbValues[i] = smoothingFactor*db + (1-smoothingFactor)*smoothedDbValues[i]

		if smoothedDbValues[i] >= peakDbValues[i] {
			peakDbValues[i] = smoothedDbValues[i]
			peakHoldCounters[i] = peakHoldFrames
		} else {
			if peakHoldCounters[i] > 0 {
				peakHoldCounters[i]--
			} else {
				peakDbValues[i] -= peakDecayRate
				if peakDbValues[i] < smoothedDbValues[i] {
					peakDbValues[i] = smoothedDbValues[i]
				}
			}
		}
	}
}

func calculateLogFreqBins(numBars int, minFreq, maxFreq float64, sampleRate, fftSize int) []logFreqBin {
	bins := make([]logFreqBin, numBars)
	
	minLogFreq := math.Log10(minFreq)
	maxLogFreq := math.Log10(maxFreq)
	logRange := maxLogFreq - minLogFreq
	
	fftBinCount := fftSize/2 + 1
	freqResolution := float64(sampleRate) / float64(fftSize)
	
	currentLogFreq := minLogFreq
	logStep := logRange / float64(numBars)
	
	for i := 0; i < numBars; i++ {
		lowerFreq := math.Pow(10, currentLogFreq)
		upperFreq := math.Pow(10, currentLogFreq+logStep)
		centerFreq := math.Sqrt(lowerFreq * upperFreq)  
		
		startBin := int(math.Ceil(lowerFreq / freqResolution))
		endBin := int(math.Floor(upperFreq / freqResolution)) + 1
		
		startBin = max(0, min(startBin, fftBinCount-1))
		endBin = max(startBin+1, min(endBin, fftBinCount))
		
		bins[i] = logFreqBin{
			centerFreq: centerFreq,
			startBin:   startBin,
			endBin:     endBin,
		}
		
		currentLogFreq += logStep
	}
	
	return bins
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}