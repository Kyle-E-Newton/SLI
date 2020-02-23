import React from 'react'
import Webcam from "react-webcam";

export default class WebcamCapture extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            screenshot: null,
            tab: 0,
            start: false,
            intervalId: null
        };
    }
    
    setRef = webcam => {
        this.webcam = webcam;
    };
      
    capture = () => {
        const screenshot = this.webcam.getScreenshot();
        this.setState({ screenshot })
    };

    timer = () => {
        if(this.start){
            this.intervalId = setInterval(() => {
                this.capture();
            }, 2000);
        }else{
            clearInterval(this.intervalId);
        }
    }

    startOn = () => {
        this.start = true;
        this.timer();
    }

    startOff = () => {
        this.start = false;
        this.timer();
    }
      
    render() {
    const videoConstraints = {
        width: 1280,
        height: 720,
        facingMode: "user"
        };
      
    return (
        <div>
            <Webcam id="cam"
                audio={false}
                ref={this.setRef}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                />
            <button onClick={this.startOn}>Start</button>
            <button onClick={this.startOff}>Stop</button>
            {this.state.screenshot ? <img src ={this.state.screenshot} /> : null}

        </div>
        );
    }
}

